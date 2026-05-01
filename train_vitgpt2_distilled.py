"""
ViT-GPT2 Distillation Training Script for WebGPU Browser Extension
6-Frame Grid Version (3x2 layout, frames 0,3,6,9,12,15 from original GIF)

Trains ViT-GPT2 on LLaVA-generated captions for GIF accessibility.
Uses VisionEncoderDecoderModel which has FULL ONNX export support.

After training, export with:
  optimum-cli export onnx --model ./vitgpt2_distilled_6frames/final --task image-to-text-with-past ./vitgpt2_onnx/

Run in Google Colab with A100 GPU.
"""

# ============================================================
# INSTALL (run in Colab)
# ============================================================
# !pip install transformers peft accelerate pillow datasets -q

import os
import json
import csv
import time
from datetime import datetime, timedelta
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

# Data - same LLaVA distillation data as SmolVLM
DISTILL_DATA = "/content/drive/MyDrive/Thesis/gif-caption-model/data/distillation_data_v2_clean.json"
OUTPUT_DIR   = "/content/drive/MyDrive/Thesis/gif-caption-model/vitgpt2_distilled_6frames"

# Grid paths - 6-frame grids (3x2 layout)
GRID_DIR_OLD = "/content/drive/MyDrive/Thesis/gif-grids"
GRID_DIR_NEW = "/content/drive/MyDrive/Thesis/gif-grids-6frames"

# Training
NUM_EPOCHS        = 3
BATCH_SIZE        = 16
GRAD_ACCUM_STEPS  = 1       # Effective batch size = 16
LEARNING_RATE     = 1e-4    # Higher LR for fewer epochs
MAX_TARGET_LENGTH = 32      # Shorter captions = faster
VAL_SPLIT         = 0.05

# LoRA - GPT2 decoder + ViT encoder attention
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
# GPT2: c_attn (self-attn QKV), c_proj (output), q_attn (cross-attn query)
# ViT:  query, value (attention projections - learns to read the 3x2 grid)
LORA_TARGET_MODULES = ["c_attn", "c_proj", "q_attn", "query", "value"]

# 6-frame grid params (must match grid generation script)
GRID_FRAMES   = 6
GRID_LAYOUT   = "3x2"
FRAME_INDICES = [0, 3, 6, 9, 12, 15]


# ============================================================
# STEP 0: Generate 6-frame grids if not already done
# ============================================================

def generate_6frame_grids(data):
    """Generate 6-frame grids from existing 16-frame grids."""
    from tqdm import tqdm

    ORIG_ROWS, ORIG_COLS = 4, 4
    ORIG_CELL  = 128
    ORIG_PAD   = 4
    ORIG_FINAL = 512

    NEW_ROWS, NEW_COLS = 2, 3
    NEW_CELL   = 128
    NEW_PAD    = 4
    NEW_FINAL  = (512, 512)
    PAD_COLOR  = (0, 0, 0)

    os.makedirs(GRID_DIR_NEW, exist_ok=True)

    def extract_frame_from_grid(grid_img, frame_idx):
        raw_w  = ORIG_COLS * ORIG_CELL + (ORIG_COLS - 1) * ORIG_PAD
        raw_h  = ORIG_ROWS * ORIG_CELL + (ORIG_ROWS - 1) * ORIG_PAD
        scale  = min(ORIG_FINAL / raw_w, ORIG_FINAL / raw_h)
        scaled_w = int(raw_w * scale)
        scaled_h = int(raw_h * scale)
        offset_x = (ORIG_FINAL - scaled_w) // 2
        offset_y = (ORIG_FINAL - scaled_h) // 2
        row = frame_idx // ORIG_COLS
        col = frame_idx % ORIG_COLS
        x  = offset_x + round(col * (ORIG_CELL + ORIG_PAD) * scale)
        y  = offset_y + round(row * (ORIG_CELL + ORIG_PAD) * scale)
        cw = round(ORIG_CELL * scale)
        ch = round(ORIG_CELL * scale)
        return grid_img.crop((x, y, x + cw, y + ch)).resize((NEW_CELL, NEW_CELL), Image.BILINEAR)

    saved, skipped = 0, 0
    for item in tqdm(data, desc="Generating 6-frame grids"):
        src_path = item["grid_path"]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(GRID_DIR_NEW, filename)

        if os.path.exists(dst_path):
            skipped += 1
            continue

        if not os.path.exists(src_path):
            continue

        try:
            orig = Image.open(src_path).convert("RGB")
            frames = [extract_frame_from_grid(orig, i) for i in FRAME_INDICES]

            grid_w = NEW_COLS * NEW_CELL + (NEW_COLS - 1) * NEW_PAD
            grid_h = NEW_ROWS * NEW_CELL + (NEW_ROWS - 1) * NEW_PAD
            grid = Image.new("RGB", (grid_w, grid_h), color=PAD_COLOR)
            for k, frame in enumerate(frames):
                r, c = divmod(k, NEW_COLS)
                grid.paste(frame, (c * (NEW_CELL + NEW_PAD), r * (NEW_CELL + NEW_PAD)))

            # Letterbox to 512x512
            img = grid.copy()
            img.thumbnail(NEW_FINAL, Image.BILINEAR)
            result = Image.new("RGB", NEW_FINAL, color=PAD_COLOR)
            result.paste(img, ((NEW_FINAL[0] - img.width) // 2, (NEW_FINAL[1] - img.height) // 2))
            result.save(dst_path)
            saved += 1
        except:
            pass

    print(f"  6-frame grids: saved={saved}, skipped={skipped}")


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
            "architecture": "VisionEncoderDecoderModel (ViT + GPT-2)",
            "onnx_exportable": True,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "max_target_length": MAX_TARGET_LENGTH,
            "val_split": VAL_SPLIT,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "grid_frames": GRID_FRAMES,
            "grid_layout": GRID_LAYOUT,
            "distillation_teacher": "LLaVA",
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
            summary["best_eval_step"]  = self.eval_losses[best_idx]["step"]
            summary["best_eval_epoch"] = self.eval_losses[best_idx]["epoch"]
        if summary["initial_train_loss"] and summary["final_train_loss"]:
            summary["train_loss_reduction"] = round(
                (summary["initial_train_loss"] - summary["final_train_loss"])
                / summary["initial_train_loss"] * 100, 2)
        if summary["initial_eval_loss"] and summary["final_eval_loss"]:
            summary["eval_loss_reduction"] = round(
                (summary["initial_eval_loss"] - summary["final_eval_loss"])
                / summary["initial_eval_loss"] * 100, 2)
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
            eval_dict  = {e["step"]: e for e in self.eval_losses}
            for step in sorted(set(train_dict.keys()) | set(eval_dict.keys())):
                writer.writerow([
                    step,
                    train_dict.get(step, eval_dict.get(step, {})).get("epoch", ""),
                    train_dict.get(step, {}).get("loss", ""),
                    eval_dict.get(step, {}).get("loss", ""),
                ])

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

class GIFCaptionDataset(Dataset):
    def __init__(self, data, feature_extractor, tokenizer, max_target_length=64):
        self.data = data
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Remap path to 6-frame grid folder
        grid_path = item['grid_path'].replace(GRID_DIR_OLD, GRID_DIR_NEW)

        try:
            image = Image.open(grid_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {grid_path}: {e}")
            return None

        caption = item['llava_caption']

        pixel_values = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("VIT-GPT2 DISTILLATION — 6-FRAME GRIDS (3x2)")
    print("LLaVA teacher → ViT-GPT2 student")
    print("Architecture: VisionEncoderDecoderModel (ONNX exportable)")
    print("=" * 60)

    logger = ResearchLogger(OUTPUT_DIR)
    logger.log_system_info()

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ──
    print(f"\nLoading: {DISTILL_DATA}")
    with open(DISTILL_DATA, 'r') as f:
        all_data = json.load(f)
    print(f"Total samples: {len(all_data):,}")

    # ── Generate 6-frame grids if needed ──
    print(f"\nChecking 6-frame grids...")
    missing = sum(
        1 for item in all_data
        if not os.path.exists(item['grid_path'].replace(GRID_DIR_OLD, GRID_DIR_NEW))
    )
    if missing > 0:
        print(f"  {missing} grids missing, generating...")
        generate_6frame_grids(all_data)
    else:
        print(f"  All 6-frame grids found in {GRID_DIR_NEW}")

    # ── Split ──
    val_size = int(len(all_data) * VAL_SPLIT)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    logger.log_config(
        train_samples=len(train_data),
        val_samples=len(val_data),
        total_samples=len(all_data)
    )

    # ── Load model ──
    print(f"\nLoading: {BASE_MODEL}")
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
    feature_extractor = ViTImageProcessor.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Generation params go in generation_config, not model.config
    model.generation_config.max_length = MAX_TARGET_LENGTH
    model.generation_config.num_beams = 4
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # ── LoRA ──
    print(f"\nApplying LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    print(f"Targets: {LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.config["total_parameters"] = total_params
    logger.config["trainable_parameters"] = trainable_params
    logger.config["trainable_percentage"] = round(trainable_params / total_params * 100, 4)

    # ── Datasets ──
    print("\nCreating datasets...")
    train_dataset = GIFCaptionDataset(train_data, feature_extractor, tokenizer, MAX_TARGET_LENGTH)
    val_dataset = GIFCaptionDataset(val_data, feature_extractor, tokenizer, MAX_TARGET_LENGTH)

    # ── Training ──
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
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",

        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
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

    # ── Save merged model (ready for ONNX export) ──
    print("\nMerging LoRA weights...")
    merged_model = model.merge_and_unload()

    final_path = f"{OUTPUT_DIR}/final"
    merged_model.save_pretrained(final_path)
    feature_extractor.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    log_data = logger.save_all()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved:       {final_path}")
    print(f"Training time:     {log_data['training_time_formatted']}")
    print(f"Final train loss:  {log_data['summary']['final_train_loss']:.4f}")
    print(f"Final eval loss:   {log_data['summary']['final_eval_loss']:.4f}")
    print(f"Best eval loss:    {log_data['summary']['best_eval_loss']:.4f}")
    print()
    print("NEXT STEPS:")
    print(f"  1. Export to ONNX:")
    print(f"     optimum-cli export onnx --model {final_path} --task image-to-text-with-past ./vitgpt2_onnx/")
    print(f"  2. Push to HuggingFace:")
    print(f"     huggingface-cli upload Patricijia/vitgpt2-gif-descriptor {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
