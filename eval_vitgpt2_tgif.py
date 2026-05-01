"""
ViT-GPT2 (TGIF-trained) Evaluation on TGIF Test Set
Same evaluation as CNN-LSTM: pycocoevalcap, official test split, filtered broken GIFs.

Run in Google Colab.
"""

import os
import json
import subprocess
import sys
import time
import numpy as np

subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "pycocoevalcap", "pycocotools", "-q"])
import nltk
for pkg in ["wordnet", "punkt", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

import torch
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH   = "/content/drive/MyDrive/Thesis/gif-caption-model/vitgpt2_tgif_6frames/final"
DRIVE_DIR    = "/content/drive/MyDrive/Thesis/cnn-lstm-training"
TGIF_TSV     = os.path.join(DRIVE_DIR, "tgif-v1.0.tsv")
TEST_SPLIT   = os.path.join(DRIVE_DIR, "test.txt")
GRID_DIR     = os.path.join(DRIVE_DIR, "grids-6frame-test")
GIF_DIR      = "/content/local_data/eval_gifs_vit"
OUTPUT_PATH  = "/content/drive/MyDrive/Thesis/gif-caption-model/vitgpt2_tgif_6frames/eval_results.json"

MAX_LENGTH   = 20
NUM_BEAMS    = 4

# Grid params
GRID_ROWS, GRID_COLS = 2, 3
CELL_SIZE    = 128
GRID_PAD     = 4
GRID_FINAL   = (512, 512)
FRAME_INDICES = [0, 3, 6, 9, 12, 15]


# ============================================================
# GRID GENERATION WITH FILTERING
# ============================================================

def make_grid_from_gif(gif_path):
    """Extract 6 frames, build 3x2 grid, filter broken GIFs."""
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

        # Check pixel variance
        sample = np.array(all_frames[0].resize((64, 64)))
        if sample.std() < 15:
            return None

        # Select 16 evenly spaced, then pick 6
        step = max(1, (len(all_frames)-1)/15) if len(all_frames) > 1 else 1
        frames16 = [all_frames[min(int(i*step), len(all_frames)-1)] for i in range(16)]
        frames6 = [frames16[i] for i in FRAME_INDICES if i < len(frames16)]
        while len(frames6) < 6:
            frames6.append(frames6[-1])

        gw = GRID_COLS * CELL_SIZE + (GRID_COLS-1) * GRID_PAD
        gh = GRID_ROWS * CELL_SIZE + (GRID_ROWS-1) * GRID_PAD
        grid = Image.new("RGB", (gw, gh), (0, 0, 0))
        for k, frame in enumerate(frames6[:6]):
            r, c = divmod(k, GRID_COLS)
            resized = frame.resize((CELL_SIZE, CELL_SIZE), Image.BILINEAR)
            grid.paste(resized, (c*(CELL_SIZE+GRID_PAD), r*(CELL_SIZE+GRID_PAD)))

        result = Image.new("RGB", GRID_FINAL, (0, 0, 0))
        grid.thumbnail(GRID_FINAL, Image.BILINEAR)
        result.paste(grid, ((GRID_FINAL[0]-grid.width)//2, (GRID_FINAL[1]-grid.height)//2))
        return result
    except:
        return None


def download_and_grid(item):
    """Download GIF, filter, build grid."""
    gif_id = item['gif_id']
    grid_path = os.path.join(GRID_DIR, f"{gif_id}.png")

    if os.path.exists(grid_path):
        return (item, grid_path)

    gif_path = os.path.join(GIF_DIR, f"{gif_id}.gif")
    try:
        if not os.path.exists(gif_path):
            import urllib.request
            urllib.request.urlretrieve(item['url'], gif_path)

        grid = make_grid_from_gif(gif_path)
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
# MAIN
# ============================================================

print("=" * 60)
print("VIT-GPT2 (TGIF-TRAINED) EVALUATION")
print("Same eval as CNN-LSTM (pycocoevalcap, filtered)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# ── 1. Load model ──
print("\n1. Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.to(device).eval()
print(f"   Model loaded on {device}")

# ── 2. Load TGIF test data ──
print("\n2. Loading TGIF test data...")

if not os.path.exists(TEST_SPLIT):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/test.txt",
        TEST_SPLIT)

with open(TEST_SPLIT) as f:
    test_urls = set(line.strip() for line in f)

url_to_captions = {}
url_to_gif_id = {}
with open(TGIF_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            url = parts[0].strip()
            caption = parts[1].strip()
            gif_id = url.split('/')[-1].replace('.gif', '')
            if url not in url_to_captions:
                url_to_captions[url] = []
            url_to_captions[url].append(caption)
            url_to_gif_id[url] = gif_id

test_data = [{'url': url, 'gif_id': url_to_gif_id[url], 'captions': url_to_captions[url]}
             for url in test_urls if url in url_to_captions]
print(f"   Test GIFs: {len(test_data):,}")

# ── 3. Generate test grids (with filtering) ──
print(f"\n3. Generating test grids (parallel, filtered)...")
os.makedirs(GRID_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

valid_test = []
failed = 0
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(download_and_grid, item): item for item in test_data}
    for future in tqdm(as_completed(futures), total=len(futures), desc="   Grids"):
        r = future.result()
        if r:
            valid_test.append(r)
        else:
            failed += 1

print(f"   Valid: {len(valid_test):,} | Filtered: {failed:,}")

# ── 4. Generate captions ──
print(f"\n4. Generating captions...")

gts = {}
res = {}
count = 0
for item, grid_path in tqdm(valid_test, desc="   Captioning"):
    try:
        img = Image.open(grid_path).convert('RGB')
        pv = processor(images=img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pv, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, no_repeat_ngram_size=3)

        caption = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        idx = str(count)
        gts[idx] = item['captions']
        res[idx] = [caption]
        count += 1
    except:
        pass

print(f"   Generated {count:,} captions")

# ── 5. Compute metrics ──
print(f"\n5. Computing metrics (pycocoevalcap)...")

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

metrics = {}
for scorer, method in [
    (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
    (Rouge(), "ROUGE-L"),
    (Cider(), "CIDEr"),
    (Meteor(), "METEOR"),
]:
    try:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                metrics[m] = round(float(s), 4)
                print(f"   {m}: {metrics[m]}")
        else:
            metrics[method] = round(float(score), 4)
            print(f"   {method}: {metrics[method]}")
    except Exception as e:
        print(f"   {method} failed: {e}")

# ── 6. Save & compare ──
report = {
    "model": "ViT-GPT2 TGIF-trained (6-frame grid, human captions)",
    "base_model": "nlpconnect/vit-gpt2-image-captioning",
    "training_data": "TGIF 80K train split (human captions)",
    "num_test_samples": count,
    "grid": "6 frames, 3x2",
    "evaluation": "vs all human references (COCO standard), filtered broken GIFs",
    "metrics": metrics,
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(report, f, indent=2)

b1 = metrics.get('BLEU-1', 0)
rl = metrics.get('ROUGE-L', 0)
mt = metrics.get('METEOR', 0)
ci = metrics.get('CIDEr', 0)

print(f"\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
print(f"{'Model':<35} {'BLEU-1':<8} {'ROUGE-L':<10} {'METEOR':<10} {'CIDEr':<10}")
print("-" * 70)
print(f"{'Apoorva CNN-LSTM (original)':<35} {'0.530':<8} {'0.390':<10} {'0.203':<10} {'0.309':<10}")
print(f"{'CNN-LSTM Retrained (ours)':<35} {'0.490':<8} {'0.384':<10} {'0.169':<10} {'0.249':<10}")
print(f"{'SmolVLM Distilled':<35} {'0.468':<8} {'0.345':<10} {'0.212':<10} {'0.375':<10}")
print(f"{'ViT-GPT2 Base':<35} {'0.338':<8} {'0.316':<10} {'0.159':<10} {'0.134':<10}")
print(f"{'ViT-GPT2 Distilled (LLaVA)':<35} {'0.323':<8} {'0.264':<10} {'---':<10} {'0.111':<10}")
print(f"{'ViT-GPT2 TGIF-trained (ours)':<35} {b1:<8} {rl:<10} {mt:<10} {ci:<10}")
print("=" * 70)
