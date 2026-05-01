"""
ViT-GPT2 Training on TGIF Human Captions (6-Frame Grids)
Same data as CNN-LSTM (80K TGIF train split) but using ViT-GPT2 architecture.

Uses 6-frame grids (3x2 layout) generated on-the-fly from GIF URLs.
ONNX exportable for browser deployment.

Run in Google Colab with GPU.
"""

import os
import json
import time
import csv
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "nlpconnect/vit-gpt2-image-captioning"

# Data
DRIVE_DIR    = "/content/drive/MyDrive/Thesis/cnn-lstm-training"
TGIF_TSV     = os.path.join(DRIVE_DIR, "tgif-v1.0.tsv")
TRAIN_SPLIT  = os.path.join(DRIVE_DIR, "train.txt")
TEST_SPLIT   = os.path.join(DRIVE_DIR, "test.txt")
GRID_DIR     = os.path.join(DRIVE_DIR, "grids-6frame")  # 6-frame grids for TGIF
OUTPUT_DIR   = "/content/drive/MyDrive/Thesis/gif-caption-model/vitgpt2_tgif_6frames"
GIF_DIR      = "/content/local_data/gifs"  # Local SSD for downloads

# Training
NUM_EPOCHS        = 3
BATCH_SIZE        = 16
GRAD_ACCUM_STEPS  = 1
LEARNING_RATE     = 1e-4
MAX_TARGET_LENGTH = 20     # TGIF captions are short
VAL_SPLIT         = 0.05

# LoRA
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj", "q_attn", "query", "value"]

# Grid
NUM_GIF_FRAMES = 16   # Extract 16 frames from GIF
GRID_FRAMES    = 6    # Select 6 for grid
FRAME_INDICES  = [0, 3, 6, 9, 12, 15]
GRID_ROWS, GRID_COLS = 2, 3
CELL_SIZE      = 128
GRID_PAD       = 4
GRID_FINAL     = (512, 512)


# ============================================================
# GRID GENERATION
# ============================================================

def make_6frame_grid(gif_path):
    """Download GIF, extract 6 frames, build 3x2 grid (512x512)."""
    try:
        img = Image.open(gif_path)
        all_frames = []
        try:
            while True:
                all_frames.append(img.convert('RGB').copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if len(all_frames) < 1:
            return None

        # Select 16 evenly spaced, then pick 6
        step = max(1, (len(all_frames)-1)/15) if len(all_frames) > 1 else 1
        frames16 = [all_frames[min(int(i*step), len(all_frames)-1)] for i in range(16)]
        frames6 = [frames16[i] for i in FRAME_INDICES if i < len(frames16)]
        while len(frames6) < 6:
            frames6.append(frames6[-1])

        # Build grid
        gw = GRID_COLS * CELL_SIZE + (GRID_COLS-1) * GRID_PAD
        gh = GRID_ROWS * CELL_SIZE + (GRID_ROWS-1) * GRID_PAD
        grid = Image.new("RGB", (gw, gh), (0, 0, 0))
        for k, frame in enumerate(frames6[:6]):
            r, c = divmod(k, GRID_COLS)
            resized = frame.resize((CELL_SIZE, CELL_SIZE), Image.BILINEAR)
            grid.paste(resized, (c*(CELL_SIZE+GRID_PAD), r*(CELL_SIZE+GRID_PAD)))

        # Letterbox to 512x512
        result = Image.new("RGB", GRID_FINAL, (0, 0, 0))
        grid.thumbnail(GRID_FINAL, Image.BILINEAR)
        result.paste(grid, ((GRID_FINAL[0]-grid.width)//2, (GRID_FINAL[1]-grid.height)//2))
        return result
    except:
        return None


def download_and_make_grid(item):
    """Download GIF, validate, and create grid. Returns (item, grid_path) or None."""
    gif_id = item['gif_id']
    grid_path = os.path.join(GRID_DIR, f"{gif_id}.png")

    if os.path.exists(grid_path):
        return (item, grid_path)

    gif_path = os.path.join(GIF_DIR, f"{gif_id}.gif")
    try:
        if not os.path.exists(gif_path):
            import urllib.request
            urllib.request.urlretrieve(item['url'], gif_path)

        # Filter broken/placeholder GIFs
        file_size = os.path.getsize(gif_path)
        if file_size < 5000:  # < 5KB = placeholder/error GIF
            os.remove(gif_path)
            return None

        # Check frame count before making grid
        img = Image.open(gif_path)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if frame_count < 2:  # Static error image
            os.remove(gif_path)
            return None

        # Check pixel variance (low = "content not available" placeholder)
        import numpy as np
        img = Image.open(gif_path).convert('RGB').resize((64, 64))
        if np.array(img).std() < 15:
            os.remove(gif_path)
            return None

        grid = make_6frame_grid(gif_path)
        try:
            os.remove(gif_path)
        except:
            pass

        if grid:
            grid.save(grid_path)
            return (item, grid_path)
    except:
        pass
    return None


# ============================================================
# DATASET
# ============================================================

class TGIFGridDataset(Dataset):
    def __init__(self, data, feature_extractor, tokenizer, max_target_length=20):
        self.data = data
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item['grid_path']).convert('RGB')
        except:
            return None

        pixel_values = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.tokenizer(
            item['caption'],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {'pixel_values': pixel_values, 'labels': labels}


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


# ============================================================
# LOGGER
# ============================================================

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and logs.get('eval_loss'):
            self.eval_losses.append({'step': state.global_step, 'loss': logs['eval_loss']})
            print(f"Step {state.global_step}: eval={logs['eval_loss']:.4f}")
        if logs and logs.get('loss'):
            self.train_losses.append({'step': state.global_step, 'loss': logs['loss']})


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("VIT-GPT2 TRAINING ON TGIF HUMAN CAPTIONS (6-FRAME GRIDS)")
    print("Full TGIF train split (80K) — same data as CNN-LSTM")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRID_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)

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

    # Download splits if needed
    if not os.path.exists(TRAIN_SPLIT):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/train.txt", TRAIN_SPLIT)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/test.txt", TEST_SPLIT)

    with open(TRAIN_SPLIT) as f:
        train_urls = set(line.strip() for line in f)

    data = [d for d in all_data if d['url'] in train_urls]
    print(f"   Total: {len(all_data):,} | Train split: {len(data):,}")

    # ── 2. Generate 6-frame grids ──
    print(f"\n2. Generating 6-frame grids (parallel download)...")

    # Check cached
    cached = sum(1 for d in data if os.path.exists(os.path.join(GRID_DIR, f"{d['gif_id']}.png")))
    to_process = [d for d in data if not os.path.exists(os.path.join(GRID_DIR, f"{d['gif_id']}.png"))]
    print(f"   Cached: {cached:,} | To process: {len(to_process):,}")

    if to_process:
        valid_results = []
        failed = 0
        t0 = time.time()
        BATCH = 256

        for batch_start in range(0, len(to_process), BATCH):
            batch = to_process[batch_start:batch_start+BATCH]
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(download_and_make_grid, item): item for item in batch}
                for future in as_completed(futures):
                    r = future.result()
                    if r:
                        valid_results.append(r)
                    else:
                        failed += 1

            done = len(valid_results) + failed
            if done % 1000 < BATCH:
                elapsed = time.time() - t0
                rate = len(valid_results) / elapsed if elapsed > 0 else 0
                remaining = (len(to_process) - done) / rate if rate > 0 else 0
                print(f"   Done: {len(valid_results):,} | Failed: {failed:,} | Rate: {rate:.1f}/s | ETA: {timedelta(seconds=int(remaining))}")

        print(f"   Generated: {len(valid_results):,} | Failed: {failed:,}")

    # Build final dataset with grid paths
    valid_data = []
    for d in data:
        grid_path = os.path.join(GRID_DIR, f"{d['gif_id']}.png")
        if os.path.exists(grid_path):
            valid_data.append({**d, 'grid_path': grid_path})

    print(f"   Total with grids: {len(valid_data):,}")

    # Split
    val_size = int(len(valid_data) * VAL_SPLIT)
    train_data = valid_data[val_size:]
    val_data = valid_data[:val_size]
    print(f"   Train: {len(train_data):,} | Val: {len(val_data):,}")

    # ── 3. Load model ──
    print(f"\n3. Loading {BASE_MODEL}...")
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
    feature_extractor = ViTImageProcessor.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.max_length = MAX_TARGET_LENGTH
    model.generation_config.num_beams = 4
    model.generation_config.no_repeat_ngram_size = 3

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {total_params:,}")

    # ── 4. LoRA ──
    print(f"\n4. Applying LoRA (r={LORA_R})...")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Datasets ──
    print("\n5. Creating datasets...")
    train_dataset = TGIFGridDataset(train_data, feature_extractor, tokenizer, MAX_TARGET_LENGTH)
    val_dataset = TGIFGridDataset(val_data, feature_extractor, tokenizer, MAX_TARGET_LENGTH)

    # ── 6. Train ──
    logger = LoggingCallback()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
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
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to="none",
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=collate_fn, callbacks=[logger],
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    trainer.train()

    # ── 7. Save ──
    print("\nMerging LoRA and saving...")
    for attr in ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']:
        if hasattr(model.config, attr):
            delattr(model.config, attr)

    merged = model.merge_and_unload()
    final_path = f"{OUTPUT_DIR}/final"
    merged.save_pretrained(final_path)
    feature_extractor.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save log
    log = {
        "model": BASE_MODEL,
        "training_data": "TGIF 80K train split (human captions)",
        "grid": "6 frames, 3x2",
        "epochs": NUM_EPOCHS,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_losses": logger.train_losses,
        "eval_losses": logger.eval_losses,
    }
    with open(f"{OUTPUT_DIR}/training_log.json", 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved to {final_path}")
    print(f"\nNext: Evaluate and export to ONNX")
    print(f"  optimum-cli export onnx --model {final_path} --task image-to-text-with-past {OUTPUT_DIR}/onnx/")


if __name__ == "__main__":
    main()
