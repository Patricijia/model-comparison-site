"""
SmolVLM-256M Training on TGIF Human Captions (Full Dataset)
Same data as CNN-LSTM and ViT-GPT2: 80K TGIF train split with official splits.

Generates 16-frame 4x4 grids (224px per frame) on-the-fly.
Grids cached on LOCAL Colab SSD (not Drive) to save storage.
Only the final model is saved to Drive.

Output:
- smolvlm_tgif/final/              - Merged model (on Drive)
- smolvlm_tgif/training_log.json   - Metrics for thesis (on Drive)
- smolvlm_tgif/loss_history.csv    - Loss data (on Drive)

Run in Google Colab with A100 GPU.
"""

# ============================================================
# INSTALL (run in Colab)
# ============================================================
# !pip install transformers peft bitsandbytes accelerate pillow -q
# !pip uninstall pillow -y && pip install pillow==10.4.0 -q

import os
import json
import csv
import time
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

# Model
BASE_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Data paths
DRIVE_DIR    = "/content/drive/MyDrive/Thesis/cnn-lstm-training"
TGIF_TSV     = os.path.join(DRIVE_DIR, "tgif-v1.0.tsv")
TRAIN_SPLIT  = os.path.join(DRIVE_DIR, "train.txt")
TEST_SPLIT   = os.path.join(DRIVE_DIR, "test.txt")

# LOCAL Colab SSD — fast, free, ~100GB available (lost on session end)
LOCAL_GRID_DIR = "/content/local_grids"
LOCAL_GIF_DIR  = "/content/local_gifs"

# Only final model goes to Drive
OUTPUT_DIR   = "/content/drive/MyDrive/Thesis/gif-caption-model/smolvlm_tgif"
# Checkpoints on local disk (not Drive)
CHECKPOINT_DIR = "/content/smolvlm_checkpoints"

# Training
NUM_EPOCHS        = 3
BATCH_SIZE        = 4
GRAD_ACCUM_STEPS  = 4       # Effective batch size = 16
LEARNING_RATE     = 2e-4
MAX_LENGTH        = 2048
VAL_SPLIT         = 0.05

# LoRA
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Grid: 16 frames, 4x4, 224px each = 908x908
GRID_ROWS, GRID_COLS = 4, 4
CELL_SIZE    = 224
GRID_PAD     = 4
GRID_FINAL   = (908, 908)
NUM_FRAMES   = 16

# Prompt
PROMPT = (
    "These frames are from an animated GIF, ordered left to right over time. "
    "Describe what is happening in a short, simple sentence."
)


# ============================================================
# GRID GENERATION + FILTERING
# ============================================================

def make_16frame_grid(gif_path):
    """Extract 16 frames from GIF, build 4x4 grid (908x908). Returns None if broken."""
    try:
        file_size = os.path.getsize(gif_path)
        if file_size < 5000:
            return None

        img = Image.open(gif_path)
        all_frames = []
        try:
            while True:
                all_frames.append(img.convert('RGB').copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if len(all_frames) < 2:
            return None

        sample = np.array(all_frames[0].resize((64, 64)))
        if sample.std() < 15:
            return None

        step = max(1, (len(all_frames) - 1) / (NUM_FRAMES - 1)) if len(all_frames) > 1 else 1
        frames = [all_frames[min(int(i * step), len(all_frames) - 1)] for i in range(NUM_FRAMES)]
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])

        gw = GRID_COLS * CELL_SIZE + (GRID_COLS - 1) * GRID_PAD
        gh = GRID_ROWS * CELL_SIZE + (GRID_ROWS - 1) * GRID_PAD
        grid = Image.new("RGB", (gw, gh), (0, 0, 0))
        for k, frame in enumerate(frames[:NUM_FRAMES]):
            r, c = divmod(k, GRID_COLS)
            resized = frame.resize((CELL_SIZE, CELL_SIZE), Image.BILINEAR)
            grid.paste(resized, (c * (CELL_SIZE + GRID_PAD), r * (CELL_SIZE + GRID_PAD)))

        result = Image.new("RGB", GRID_FINAL, (0, 0, 0))
        grid.thumbnail(GRID_FINAL, Image.BILINEAR)
        result.paste(grid, ((GRID_FINAL[0] - grid.width) // 2, (GRID_FINAL[1] - grid.height) // 2))
        return result
    except:
        return None


def download_and_make_grid(item):
    """Download GIF → make grid → save to LOCAL disk → delete GIF immediately."""
    gif_id = item['gif_id']
    grid_path = os.path.join(LOCAL_GRID_DIR, f"{gif_id}.png")

    if os.path.exists(grid_path):
        return (item, grid_path)

    gif_path = os.path.join(LOCAL_GIF_DIR, f"{gif_id}.gif")
    try:
        if not os.path.exists(gif_path):
            import urllib.request
            urllib.request.urlretrieve(item['url'], gif_path)

        grid = make_16frame_grid(gif_path)

        # Delete GIF immediately to save space
        try:
            os.remove(gif_path)
        except:
            pass

        if grid:
            grid.save(grid_path)
            return (item, grid_path)
    except:
        # Clean up on error too
        try:
            os.remove(gif_path)
        except:
            pass
    return None


# ============================================================
# RESEARCH LOGGER
# ============================================================

class ResearchLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.start_time = None
        self.end_time = None
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.config = {}
        self.system_info = {}

    def log_config(self, **kwargs):
        self.config = {
            "model": BASE_MODEL,
            "training_data": "TGIF human captions (official train split)",
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "val_split": VAL_SPLIT,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "prompt": PROMPT,
            **kwargs
        }

    def log_system_info(self):
        self.system_info = {
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            self.system_info["gpu_name"] = torch.cuda.get_device_name(0)
            self.system_info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )

    def start_training(self):
        self.start_time = time.time()

    def end_training(self):
        self.end_time = time.time()

    def log_step(self, step, epoch, train_loss=None, eval_loss=None, lr=None):
        if train_loss is not None:
            self.train_losses.append({"step": step, "epoch": epoch, "loss": train_loss})
        if eval_loss is not None:
            self.eval_losses.append({"step": step, "epoch": epoch, "loss": eval_loss})
        if lr is not None:
            self.learning_rates.append({"step": step, "lr": lr})

    def save_all(self):
        training_time = self.end_time - self.start_time if self.start_time and self.end_time else None

        summary = {
            "total_steps": len(self.train_losses),
            "final_train_loss": self.train_losses[-1]["loss"] if self.train_losses else None,
            "final_eval_loss": self.eval_losses[-1]["loss"] if self.eval_losses else None,
            "best_eval_loss": min(e["loss"] for e in self.eval_losses) if self.eval_losses else None,
            "initial_train_loss": self.train_losses[0]["loss"] if self.train_losses else None,
            "initial_eval_loss": self.eval_losses[0]["loss"] if self.eval_losses else None,
        }

        if self.eval_losses:
            best_idx = [e["loss"] for e in self.eval_losses].index(summary["best_eval_loss"])
            summary["best_eval_step"] = self.eval_losses[best_idx]["step"]
            summary["best_eval_epoch"] = self.eval_losses[best_idx]["epoch"]

        if summary["initial_train_loss"] and summary["final_train_loss"]:
            summary["train_loss_reduction"] = round(
                (summary["initial_train_loss"] - summary["final_train_loss"]) / summary["initial_train_loss"] * 100, 2
            )
        if summary["initial_eval_loss"] and summary["final_eval_loss"]:
            summary["eval_loss_reduction"] = round(
                (summary["initial_eval_loss"] - summary["final_eval_loss"]) / summary["initial_eval_loss"] * 100, 2
            )

        if summary["final_train_loss"] and summary["final_eval_loss"]:
            gap = summary["final_eval_loss"] - summary["final_train_loss"]
            summary["train_eval_gap"] = round(gap, 4)
            summary["overfitting"] = gap > 0.05

        log_data = {
            "config": self.config,
            "system_info": self.system_info,
            "training_time_seconds": training_time,
            "training_time_formatted": str(timedelta(seconds=int(training_time))) if training_time else None,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "learning_rates": self.learning_rates,
            "summary": summary,
            "data_info": {
                "train_samples": self.config.get("train_samples"),
                "val_samples": self.config.get("val_samples"),
                "total_samples": self.config.get("total_samples"),
                "unique_gifs": self.config.get("unique_gifs"),
            },
            "model_info": {
                "base_model": BASE_MODEL,
                "total_parameters": self.config.get("total_parameters"),
                "trainable_parameters": self.config.get("trainable_parameters"),
                "trainable_percentage": self.config.get("trainable_percentage"),
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
            }
        }

        with open(os.path.join(self.output_dir, "training_log.json"), 'w') as f:
            json.dump(log_data, f, indent=2)

        with open(os.path.join(self.output_dir, "loss_history.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss", "eval_loss"])
            train_dict = {t["step"]: t for t in self.train_losses}
            eval_dict = {e["step"]: e for e in self.eval_losses}
            all_steps = sorted(set(train_dict.keys()) | set(eval_dict.keys()))
            for step in all_steps:
                train_loss = train_dict.get(step, {}).get("loss", "")
                eval_loss = eval_dict.get(step, {}).get("loss", "")
                epoch = train_dict.get(step, eval_dict.get(step, {})).get("epoch", "")
                writer.writerow([step, epoch, train_loss, eval_loss])

        print(f"Saved: training_log.json, loss_history.csv")
        return log_data


class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.start_training()

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.end_training()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.log_step(
                step=state.global_step,
                epoch=state.epoch,
                train_loss=logs.get('loss'),
                eval_loss=logs.get('eval_loss'),
                lr=logs.get('learning_rate')
            )
            if logs.get('eval_loss'):
                print(f"Step {state.global_step}: train={logs.get('loss', 0):.4f}, eval={logs['eval_loss']:.4f}")


# ============================================================
# DATASET
# ============================================================

class TGIFSmolVLMDataset(Dataset):
    """TGIF dataset for SmolVLM. Grids loaded from local Colab SSD."""

    def __init__(self, data, processor, max_length=2048):
        self.data = data
        self.processor = processor
        self.max_length = max_length

        if len(data) > 0:
            sample = data[0]
            print(f"  Sample grid: {sample['grid_path']}")
            print(f"  Sample caption: {sample['caption'][:60]}...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item['grid_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {item['grid_path']}: {e}")
            return None

        caption = item['caption']

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is not None and image_token_id != self.processor.tokenizer.unk_token_id:
            labels[input_ids == image_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    result = {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch])
    }

    pixel_values = [b['pixel_values'] for b in batch]
    shapes = [pv.shape for pv in pixel_values]
    if len(set(shapes)) == 1:
        result['pixel_values'] = torch.stack(pixel_values)
    else:
        max_patches = max(pv.shape[0] for pv in pixel_values)
        padded = []
        for pv in pixel_values:
            if pv.shape[0] < max_patches:
                padding = torch.zeros(max_patches - pv.shape[0], *pv.shape[1:], dtype=pv.dtype)
                pv = torch.cat([pv, padding], dim=0)
            padded.append(pv)
        result['pixel_values'] = torch.stack(padded)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SMOLVLM-256M TRAINING ON TGIF HUMAN CAPTIONS")
    print("Same data as CNN-LSTM & ViT-GPT2 (80K train split)")
    print("Grids on local SSD, only final model saved to Drive")
    print("=" * 60)

    logger = ResearchLogger(OUTPUT_DIR)
    logger.log_system_info()

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: No GPU detected!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_GRID_DIR, exist_ok=True)
    os.makedirs(LOCAL_GIF_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── 1. Parse TGIF + official splits ──
    print(f"\n1. Parsing TGIF dataset...")

    all_data = []
    with open(TGIF_TSV, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                url = parts[0].strip()
                caption = parts[1].strip().lower()
                gif_id = url.split('/')[-1].replace('.gif', '')
                all_data.append({'url': url, 'caption': caption, 'gif_id': gif_id})

    print(f"   Total TGIF entries: {len(all_data):,}")

    if not os.path.exists(TRAIN_SPLIT):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/train.txt",
            TRAIN_SPLIT)
    if not os.path.exists(TEST_SPLIT):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/test.txt",
            TEST_SPLIT)

    with open(TRAIN_SPLIT) as f:
        train_urls = set(line.strip() for line in f)

    data = [d for d in all_data if d['url'] in train_urls]
    unique_gifs = len(set(d['gif_id'] for d in data))
    print(f"   Train split: {len(data):,} entries ({unique_gifs:,} unique GIFs)")

    # ── 2. Generate grids on LOCAL SSD (not Drive) ──
    print(f"\n2. Generating grids on local SSD ({LOCAL_GRID_DIR})...")
    print(f"   GIFs downloaded one-by-one, converted to grid, then deleted.")

    # Deduplicate: only one grid per unique GIF
    seen_gifs = set()
    unique_items = []
    for d in data:
        if d['gif_id'] not in seen_gifs:
            seen_gifs.add(d['gif_id'])
            unique_items.append(d)

    # Check cached
    cached = sum(1 for d in unique_items if os.path.exists(os.path.join(LOCAL_GRID_DIR, f"{d['gif_id']}.png")))
    to_process = [d for d in unique_items if not os.path.exists(os.path.join(LOCAL_GRID_DIR, f"{d['gif_id']}.png"))]
    print(f"   Cached: {cached:,} | To process: {len(to_process):,}")

    if to_process:
        generated = 0
        failed = 0
        t0 = time.time()
        # Small batches: download a few GIFs, make grids, delete GIFs, repeat
        BATCH = 64

        for batch_start in range(0, len(to_process), BATCH):
            batch = to_process[batch_start:batch_start + BATCH]
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(download_and_make_grid, item): item for item in batch}
                for future in as_completed(futures):
                    r = future.result()
                    if r:
                        generated += 1
                    else:
                        failed += 1

            # Clean up any leftover GIFs in this batch
            for f_name in os.listdir(LOCAL_GIF_DIR):
                try:
                    os.remove(os.path.join(LOCAL_GIF_DIR, f_name))
                except:
                    pass

            done = generated + failed
            if done % 500 < BATCH or batch_start + BATCH >= len(to_process):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(to_process) - done) / rate if rate > 0 else 0
                print(f"   Generated: {generated:,} | Failed: {failed:,} | "
                      f"Rate: {rate:.1f}/s | ETA: {timedelta(seconds=int(remaining))}")

        print(f"   New grids: {generated:,} | Filtered out: {failed:,}")

    # Check local disk usage
    grid_count = len([f for f in os.listdir(LOCAL_GRID_DIR) if f.endswith('.png')])
    print(f"   Grids on local SSD: {grid_count:,}")

    # ── 3. Build final dataset ──
    print(f"\n3. Building dataset...")

    valid_data = []
    filtered = 0
    for d in data:
        grid_path = os.path.join(LOCAL_GRID_DIR, f"{d['gif_id']}.png")
        if os.path.exists(grid_path):
            valid_data.append({**d, 'grid_path': grid_path})
        else:
            filtered += 1

    unique_valid = len(set(d['gif_id'] for d in valid_data))
    print(f"   Valid samples: {len(valid_data):,} ({unique_valid:,} unique GIFs)")
    print(f"   Filtered (broken/placeholder): {filtered:,}")

    import random
    random.seed(42)
    random.shuffle(valid_data)

    val_size = int(len(valid_data) * VAL_SPLIT)
    train_data = valid_data[val_size:]
    val_data = valid_data[:val_size]

    print(f"   Train: {len(train_data):,} | Val: {len(val_data):,}")

    logger.log_config(
        train_samples=len(train_data),
        val_samples=len(val_data),
        total_samples=len(valid_data),
        unique_gifs=unique_valid,
        filtered_samples=filtered,
    )

    # ── 4. Load model ──
    print(f"\n4. Loading {BASE_MODEL}...")

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    total_params = model.num_parameters()
    print(f"   Total params: {total_params:,}")

    # ── 5. Apply LoRA ──
    print(f"\n5. Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.config["total_parameters"] = total_params
    logger.config["trainable_parameters"] = trainable_params
    logger.config["trainable_percentage"] = round(trainable_params / total_params * 100, 4)

    # ── 6. Datasets ──
    print("\n6. Creating datasets...")
    train_dataset = TGIFSmolVLMDataset(train_data, processor, MAX_LENGTH)
    val_dataset = TGIFSmolVLMDataset(val_data, processor, MAX_LENGTH)

    # ── 7. Training ──
    print(f"\n7. Training config:")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"   LR: {LEARNING_RATE}")
    print(f"   Checkpoints: local SSD (not Drive)")

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,  # Checkpoints on LOCAL disk
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=2000,           # Save less often
        save_total_limit=1,        # Keep only 1 checkpoint to save space
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=100,
        logging_dir=f"{CHECKPOINT_DIR}/logs",  # Logs on local too
        report_to="none",

        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[LoggingCallback(logger)],
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    trainer.train()

    # ── 8. Save ONLY final model to Drive ──
    print("\nMerging LoRA and saving to Drive...")

    final_path = f"{OUTPUT_DIR}/final"
    os.makedirs(final_path, exist_ok=True)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    # Save logs to Drive
    log_data = logger.save_all()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to Drive: {final_path}")
    print(f"Training time: {log_data['training_time_formatted']}")
    print(f"Final train loss: {log_data['summary']['final_train_loss']:.4f}")
    print(f"Final eval loss: {log_data['summary']['final_eval_loss']:.4f}")
    print(f"Best eval loss: {log_data['summary']['best_eval_loss']:.4f}")
    print(f"Unique GIFs: {unique_valid:,}")
    print(f"Total samples: {len(valid_data):,}")

    # Verify Drive save
    saved_files = os.listdir(final_path)
    total_size = sum(os.path.getsize(os.path.join(final_path, f)) for f in saved_files)
    print(f"\nDrive usage: {total_size / 1e6:.0f} MB ({len(saved_files)} files)")

    print("\nNext steps:")
    print("  1. Run eval_smolvlm_tgif.py")
    print("  2. Upload to HuggingFace")
    print("  3. Update browser extension model ID")
    print("=" * 60)


if __name__ == "__main__":
    main()
