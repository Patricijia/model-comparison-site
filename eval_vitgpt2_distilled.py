"""
ViT-GPT2 Distilled Evaluation on TGIF Test Set
Uses pycocoevalcap (MSCOCO library) — same as Apoorva and SmolVLM evaluations.

Evaluates on official TGIF test split (~10K GIFs) using 6-frame grids.
Scores against all human references simultaneously.

Run in Google Colab.
"""

import os
import json
import subprocess
import sys
import time

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "pycocoevalcap", "pycocotools", "-q"])
import nltk
for pkg in ["wordnet", "punkt", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

import torch
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH     = "/content/drive/MyDrive/Thesis/gif-caption-model/vitgpt2_distilled_6frames/final"
TGIF_TSV       = "/content/drive/MyDrive/Thesis/cnn-lstm-training/tgif-v1.0.tsv"
GRID_DIR       = "/content/drive/MyDrive/Thesis/gif-grids-6frames"
OUTPUT_PATH    = "/content/drive/MyDrive/Thesis/gif-caption-model/vitgpt2_distilled_6frames/eval_results.json"

# Official TGIF test split
TEST_SPLIT_URL = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/test.txt"
TEST_SPLIT     = "/content/drive/MyDrive/Thesis/cnn-lstm-training/test.txt"

MAX_LENGTH     = 16
NUM_BEAMS      = 4


# ============================================================
# STEP 1: Load model
# ============================================================

print("=" * 60)
print("VIT-GPT2 DISTILLED EVALUATION (TGIF Test Set)")
print("=" * 60)

print("\n1. Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print(f"   Model loaded on {device}")


# ============================================================
# STEP 2: Load TGIF test data
# ============================================================

print("\n2. Loading TGIF test data...")

# Download test split if needed
if not os.path.exists(TEST_SPLIT):
    import urllib.request
    urllib.request.urlretrieve(TEST_SPLIT_URL, TEST_SPLIT)

with open(TEST_SPLIT) as f:
    test_urls = set(line.strip() for line in f)
print(f"   Test URLs: {len(test_urls):,}")

# Parse TGIF TSV — build URL → captions mapping
url_to_captions = {}
with open(TGIF_TSV, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            url = parts[0].strip()
            caption = parts[1].strip()
            if url not in url_to_captions:
                url_to_captions[url] = []
            url_to_captions[url].append(caption)

# Build URL → grid filename mapping
def url_to_grid_filename(url):
    """Convert TGIF URL to grid filename (same format as grid generation script)."""
    # Grid filenames are based on the URL path
    name = url.replace("https://", "").replace("http://", "").replace("/", "-").replace(".gif", ".png")
    return name

# Find test GIFs that have both captions and grid images
test_data = []
missing_grids = 0
for url in test_urls:
    if url not in url_to_captions:
        continue
    grid_filename = url_to_grid_filename(url)
    grid_path = os.path.join(GRID_DIR, grid_filename)
    if os.path.exists(grid_path):
        test_data.append({
            'url': url,
            'grid_path': grid_path,
            'captions': url_to_captions[url],
        })
    else:
        missing_grids += 1

print(f"   Test samples with grids: {len(test_data):,}")
print(f"   Missing grids: {missing_grids:,}")

if len(test_data) == 0:
    print(f"\n   No pre-built test grids found. Generating from GIF URLs...")
    print(f"   This downloads test GIFs, extracts 6 frames, and builds grids.")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import urllib.request

    TEST_GRID_DIR = os.path.join(os.path.dirname(GRID_DIR), "gif-grids-6frames-test")
    os.makedirs(TEST_GRID_DIR, exist_ok=True)

    NEW_ROWS, NEW_COLS = 2, 3
    NEW_CELL, NEW_PAD = 128, 4
    NEW_FINAL = (512, 512)
    FRAME_INDICES_16 = [0, 3, 6, 9, 12, 15]  # 6 from 16

    def download_and_make_grid(url):
        """Download GIF, extract 6 frames, build grid."""
        grid_filename = url_to_grid_filename(url)
        grid_path = os.path.join(TEST_GRID_DIR, grid_filename)
        if os.path.exists(grid_path):
            return grid_path

        try:
            tmp_path = f"/tmp/test_{grid_filename.replace('.png', '.gif')}"
            urllib.request.urlretrieve(url, tmp_path)

            img = Image.open(tmp_path)
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
            step16 = max(1, (len(all_frames) - 1) / 15) if len(all_frames) > 1 else 1
            frames16 = [all_frames[min(int(i * step16), len(all_frames)-1)] for i in range(16)]
            frames6 = [frames16[i] for i in FRAME_INDICES_16 if i < len(frames16)]
            while len(frames6) < 6:
                frames6.append(frames6[-1])

            # Build 2x3 grid
            grid_w = NEW_COLS * NEW_CELL + (NEW_COLS - 1) * NEW_PAD
            grid_h = NEW_ROWS * NEW_CELL + (NEW_ROWS - 1) * NEW_PAD
            grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
            for k, frame in enumerate(frames6[:6]):
                r, c = divmod(k, NEW_COLS)
                resized = frame.resize((NEW_CELL, NEW_CELL), Image.BILINEAR)
                grid.paste(resized, (c * (NEW_CELL + NEW_PAD), r * (NEW_CELL + NEW_PAD)))

            # Letterbox to 512x512
            result = Image.new("RGB", NEW_FINAL, (0, 0, 0))
            grid.thumbnail(NEW_FINAL, Image.BILINEAR)
            result.paste(grid, ((NEW_FINAL[0] - grid.width) // 2, (NEW_FINAL[1] - grid.height) // 2))
            result.save(grid_path)

            try:
                os.remove(tmp_path)
            except:
                pass
            return grid_path
        except:
            return None

    # Parallel download + grid generation
    test_urls_with_captions = [url for url in test_urls if url in url_to_captions]
    print(f"   Processing {len(test_urls_with_captions):,} test GIFs (parallel)...")

    generated = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_and_make_grid, url): url for url in test_urls_with_captions}
        for future in tqdm(as_completed(futures), total=len(futures), desc="   Generating test grids"):
            url = futures[future]
            result = future.result()
            if result:
                test_data.append({
                    'url': url,
                    'grid_path': result,
                    'captions': url_to_captions[url],
                })
                generated += 1
            else:
                failed += 1

    print(f"   Generated: {generated:,} | Failed: {failed:,}")
    GRID_DIR = TEST_GRID_DIR  # Use test grid dir for this run


# ============================================================
# STEP 3: Generate captions
# ============================================================

print(f"\n3. Generating captions for {len(test_data):,} test GIFs...")

predictions = {}
t0 = time.time()

for i, item in enumerate(tqdm(test_data, desc="Captioning")):
    try:
        img = Image.open(item['grid_path']).convert('RGB')
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
            )

        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        predictions[str(i)] = caption
    except Exception as e:
        predictions[str(i)] = ""
        if i < 5:
            print(f"   Error on {i}: {e}")

elapsed = time.time() - t0
print(f"   Generated {len(predictions):,} captions in {elapsed:.0f}s ({len(predictions)/elapsed:.1f}/s)")


# ============================================================
# STEP 4: Compute metrics (MSCOCO evaluation)
# ============================================================

print(f"\n4. Computing metrics (pycocoevalcap)...")

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

# Build gts and res in COCO format
gts = {}
res = {}
for i, item in enumerate(test_data):
    idx = str(i)
    if idx in predictions and predictions[idx]:
        gts[idx] = item['captions']       # all human references
        res[idx] = [predictions[idx]]      # model prediction

print(f"   Evaluating {len(gts):,} samples with {sum(len(v) for v in gts.values()):,} total references")

results = {}
for scorer, method in [
    (Bleu(4),  ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
    (Rouge(),  "ROUGE-L"),
    (Meteor(), "METEOR"),
    (Cider(),  "CIDEr"),
]:
    try:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                results[m] = round(float(s), 4)
                print(f"   {m}: {results[m]}")
        else:
            results[method] = round(float(score), 4)
            print(f"   {method}: {results[method]}")
    except Exception as e:
        print(f"   Warning: {method} failed — {e}")


# ============================================================
# STEP 5: Save results
# ============================================================

report = {
    "model": "ViT-GPT2 Distilled (LLaVA teacher, 6-frame grid)",
    "base_model": "nlpconnect/vit-gpt2-image-captioning",
    "model_path": MODEL_PATH,
    "num_samples": len(gts),
    "num_frames": 6,
    "grid_layout": "3x2",
    "max_length": MAX_LENGTH,
    "num_beams": NUM_BEAMS,
    "evaluation": "vs all human references simultaneously (COCO standard)",
    "metrics": results,
    "comparison": {
        "apoorva_cnn_lstm": {
            "BLEU-1": 0.530, "ROUGE-L": 0.390, "METEOR": 0.203, "CIDEr": 0.309
        },
        "smolvlm_distilled": {
            "BLEU-1": 0.468, "ROUGE-L": 0.345, "METEOR": 0.212, "CIDEr": 0.375
        },
        "vitgpt2_base": {
            "BLEU-1": 0.338, "ROUGE-L": 0.316, "METEOR": 0.159, "CIDEr": 0.134
        },
    }
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n5. Results saved to {OUTPUT_PATH}")

# ============================================================
# Print comparison
# ============================================================

print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
print(f"{'Model':<30} {'BLEU-1':<8} {'ROUGE-L':<10} {'METEOR':<10} {'CIDEr':<10}")
print("-" * 70)
print(f"{'Apoorva CNN-LSTM':<30} {'0.530':<8} {'0.390':<10} {'0.203':<10} {'0.309':<10}")
print(f"{'SmolVLM Distilled':<30} {'0.468':<8} {'0.345':<10} {'0.212':<10} {'0.375':<10}")
print(f"{'ViT-GPT2 Base':<30} {'0.338':<8} {'0.316':<10} {'0.159':<10} {'0.134':<10}")
b1 = results.get('BLEU-1', 0)
rl = results.get('ROUGE-L', 0)
mt = results.get('METEOR', 0)
ci = results.get('CIDEr', 0)
print(f"{'ViT-GPT2 Distilled (ours)':<30} {b1:<8} {rl:<10} {mt:<10} {ci:<10}")
print("=" * 70)

# Delta vs base
print(f"\nImprovement over ViT-GPT2 Base:")
print(f"  BLEU-1:  {b1 - 0.338:+.4f}")
print(f"  ROUGE-L: {rl - 0.316:+.4f}")
print(f"  METEOR:  {mt - 0.159:+.4f}")
print(f"  CIDEr:   {ci - 0.134:+.4f}")
print("=" * 70)
