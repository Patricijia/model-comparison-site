"""
CNN-LSTM Evaluation on TGIF Test Set
Same evaluation as Apoorva: pycocoevalcap (MSCOCO library), official TGIF test split.

Pipeline: Load test GIF URLs → download → extract 16 frames → EfficientNet features →
BiLSTM encoder → Attention decoder → caption → score vs human references.

Run in Google Colab with A100 GPU.
"""

import os
import json
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install eval deps
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "pycocoevalcap", "pycocotools", "-q"])
import nltk
for pkg in ["wordnet", "punkt", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ============================================================
# CONFIG
# ============================================================

DRIVE_DIR    = "/content/drive/MyDrive/Thesis/cnn-lstm-training"
OUTPUT_DIR   = os.path.join(DRIVE_DIR, "output")
TGIF_TSV     = os.path.join(DRIVE_DIR, "tgif-v1.0.tsv")
FEATURE_DIR  = os.path.join(DRIVE_DIR, "features")
TEST_SPLIT   = os.path.join(DRIVE_DIR, "test.txt")
GIF_DIR      = "/content/local_data/eval_gifs"

MODEL_PATH   = os.path.join(OUTPUT_DIR, "caption_bilstm_attention_best.pt")
VOCAB_PATH   = os.path.join(OUTPUT_DIR, "word2idx.json")
EMBED_PATH   = os.path.join(OUTPUT_DIR, "embedding_matrix.npy")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "eval_results.json")

NUM_FRAMES   = 16
MAX_SEQ_LEN  = 20
ENCODER_DIM  = 1280
HIDDEN_SIZE  = 512


# ============================================================
# MODEL DEFINITION (same as training)
# ============================================================

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
    def forward(self, encoder_outputs, hidden):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, hidden), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)

class CaptionGenerator(nn.Module):
    def __init__(self, embed_matrix, encoder_dim=1280, hidden_size=512):
        super().__init__()
        num_embeddings, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.encoder_lstm = nn.LSTM(encoder_dim, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_embeddings)


def generate_caption(model, features, word2idx, idx2word, device, max_len=20):
    """Generate caption for a single GIF (greedy decoding)."""
    model.eval()
    with torch.no_grad():
        features = features.unsqueeze(0).to(device)
        encoder_outputs, (h, c) = model.encoder_lstm(features)
        h = h.sum(dim=0).unsqueeze(0)
        c = c.sum(dim=0).unsqueeze(0)

        words = []
        prev_word = "<start>"
        for _ in range(max_len):
            token = word2idx.get(prev_word, word2idx["<unk>"])
            embed = model.embedding(torch.tensor([[token]]).to(device)).squeeze(1)
            attn_weights = model.attention(encoder_outputs, h[-1])
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            lstm_input = torch.cat((embed, context), dim=1).unsqueeze(1)
            output, (h, c) = model.decoder_lstm(lstm_input, (h, c))
            scores = model.fc(output.squeeze(1))
            next_id = scores.argmax(dim=1).item()
            next_word = idx2word.get(str(next_id), "<unk>")
            if next_word == "<end>":
                break
            words.append(next_word)
            prev_word = next_word

    return " ".join(words)


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("CNN-LSTM EVALUATION ON TGIF TEST SET")
print("Same evaluation as Apoorva (pycocoevalcap)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# ── 1. Load model ──
print("\n1. Loading model...")
with open(VOCAB_PATH) as f:
    word2idx = json.load(f)
idx2word = {str(v): k for k, v in word2idx.items()}
embed_matrix = np.load(EMBED_PATH)

model = CaptionGenerator(embed_matrix, ENCODER_DIM, HIDDEN_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"   Model: {sum(p.numel() for p in model.parameters()):,} params")
print(f"   Vocab: {len(word2idx)} words")

# Load EfficientNet for feature extraction
cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
cnn.classifier = nn.Identity()
cnn.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ── 2. Load TGIF test data ──
print("\n2. Loading TGIF test data...")

# Download test split if needed
if not os.path.exists(TEST_SPLIT):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/test.txt",
        TEST_SPLIT)

with open(TEST_SPLIT) as f:
    test_urls = set(line.strip() for line in f)

# Parse TGIF TSV — URL → captions
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

# Build test set
test_data = []
for url in test_urls:
    if url in url_to_captions:
        test_data.append({
            'url': url,
            'gif_id': url_to_gif_id[url],
            'captions': url_to_captions[url],
        })

print(f"   Test GIFs: {len(test_data):,}")

# ── 3. Extract features for test GIFs ──
print(f"\n3. Extracting features for test GIFs...")
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_frames_pil(gif_path, num_frames=16):
    try:
        img = Image.open(gif_path)
        all_frames = []
        try:
            while True:
                all_frames.append(img.convert('RGB').copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if not all_frames:
            return []
        step = max(1, (len(all_frames)-1)/(num_frames-1)) if len(all_frames) > 1 else 1
        selected = [all_frames[min(int(i*step), len(all_frames)-1)] for i in range(num_frames)]
        while len(selected) < num_frames:
            selected.append(selected[-1])
        return selected[:num_frames]
    except:
        return []

def download_gif(url, path):
    if os.path.exists(path):
        return True
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
        return True
    except:
        return False

def download_one(item):
    gif_path = os.path.join(GIF_DIR, f"{item['gif_id']}.gif")
    feature_path = os.path.join(FEATURE_DIR, f"{item['gif_id']}.pt")
    if os.path.exists(feature_path):
        return (item, feature_path, None)
    if download_gif(item['url'], gif_path):
        # Filter bad GIFs
        try:
            file_size = os.path.getsize(gif_path)
            if file_size < 5000:  # Placeholder GIF
                os.remove(gif_path)
                return None
        except:
            pass
        frames = extract_frames_pil(gif_path, NUM_FRAMES)
        try: os.remove(gif_path)
        except: pass
        if frames and len(frames) >= 2:
            # Check pixel variance
            import numpy as np
            sample = np.array(frames[0].resize((64, 64)))
            if sample.std() < 15:  # "Content not available" placeholder
                return None
            return (item, None, frames)
    return None

# Parallel download
print("   Downloading test GIFs (parallel)...")
results = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(download_one, item): item for item in test_data}
    for future in tqdm(as_completed(futures), total=len(futures), desc="   Downloading"):
        r = future.result()
        if r:
            results.append(r)

# Extract features
print(f"   Extracting features for {len(results):,} GIFs...")
test_features = {}
cached = 0
computed = 0
for item, feature_path, frames in tqdm(results, desc="   Features"):
    gif_id = item['gif_id']
    fp = os.path.join(FEATURE_DIR, f"{gif_id}.pt")
    if feature_path and os.path.exists(feature_path):
        try:
            feat = torch.load(feature_path, map_location='cpu', weights_only=True)
            if feat.shape == (NUM_FRAMES, ENCODER_DIM):
                # Filter bad features (placeholder/error GIFs)
                if feat.abs().mean() < 0.01:
                    continue
                if (feat[0] - feat[-1]).abs().max() < 0.001:
                    continue
                test_features[gif_id] = feat
                cached += 1
                continue
        except: pass
    if frames:
        images = torch.stack([transform(f) for f in frames[:NUM_FRAMES]]).to(device)
        with torch.no_grad():
            feat = cnn(images).cpu()
        torch.save(feat, fp)
        test_features[gif_id] = feat
        computed += 1

print(f"   Features: {len(test_features):,} (cached: {cached}, computed: {computed})")

del cnn
torch.cuda.empty_cache()

# ── 4. Generate captions ──
print(f"\n4. Generating captions for {len(test_features):,} test GIFs...")

gts = {}
res = {}
count = 0
for item, _, _ in tqdm(results, desc="   Captioning"):
    gif_id = item['gif_id']
    if gif_id not in test_features:
        continue
    caption = generate_caption(model, test_features[gif_id], word2idx, idx2word, device, MAX_SEQ_LEN)
    idx = str(count)
    gts[idx] = item['captions']
    res[idx] = [caption]
    count += 1

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
    (Meteor(), "METEOR"),
    (Cider(), "CIDEr"),
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

# ── 6. Save results ──
report = {
    "model": "CNN-LSTM (EfficientNet-B0 + BiLSTM + Attention)",
    "training_data": "TGIF (80K train split, human captions)",
    "num_test_samples": count,
    "num_frames": NUM_FRAMES,
    "evaluation": "vs all human references simultaneously (COCO standard, same as Apoorva)",
    "metrics": metrics,
    "comparison": {
        "apoorva_original": {"BLEU-1": 0.530, "ROUGE-L": 0.390, "METEOR": 0.203, "CIDEr": 0.309},
        "smolvlm_distilled": {"BLEU-1": 0.468, "ROUGE-L": 0.345, "METEOR": 0.212, "CIDEr": 0.375},
        "vitgpt2_base": {"BLEU-1": 0.338, "ROUGE-L": 0.316, "METEOR": 0.159, "CIDEr": 0.134},
    }
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n6. Results saved to {RESULTS_PATH}")

# ── Print comparison ──
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
print(f"{'Model':<35} {'BLEU-1':<8} {'ROUGE-L':<10} {'METEOR':<10} {'CIDEr':<10}")
print("-" * 70)
print(f"{'Apoorva CNN-LSTM (original)':<35} {'0.530':<8} {'0.390':<10} {'0.203':<10} {'0.309':<10}")
print(f"{'SmolVLM Distilled':<35} {'0.468':<8} {'0.345':<10} {'0.212':<10} {'0.375':<10}")
print(f"{'ViT-GPT2 Base':<35} {'0.338':<8} {'0.316':<10} {'0.159':<10} {'0.134':<10}")
b1 = metrics.get('BLEU-1', 0)
rl = metrics.get('ROUGE-L', 0)
mt = metrics.get('METEOR', 0)
ci = metrics.get('CIDEr', 0)
print(f"{'CNN-LSTM Retrained (ours)':<35} {b1:<8} {rl:<10} {mt:<10} {ci:<10}")
print("=" * 70)

print(f"\nDelta vs Apoorva original:")
print(f"  BLEU-1:  {b1 - 0.530:+.4f}")
print(f"  ROUGE-L: {rl - 0.390:+.4f}")
print(f"  METEOR:  {mt - 0.203:+.4f}")
print(f"  CIDEr:   {ci - 0.309:+.4f}")
print("=" * 70)
