"""
CNN-LSTM Training Script for Browser Extension
EfficientNet-B0 + BiLSTM + Attention on TGIF Dataset

Trains the same architecture as the reimplemented baseline but on the
full TGIF dataset (~90K+ GIFs) to match Apoorva's original training scale.

After training, export to ONNX for browser deployment.

Run in Google Colab with A100 GPU.
"""

# ============================================================
# INSTALL (run in Colab)
# ============================================================
# !pip install torchvision pillow -q

import os
import json
import csv
import time
import subprocess
from datetime import datetime, timedelta
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

# Data - Drive for persistent storage, local SSD for temp files
DRIVE_DIR      = "/content/drive/MyDrive/Thesis/cnn-lstm-training"
TGIF_TSV       = os.path.join(DRIVE_DIR, "tgif-v1.0.tsv")
OUTPUT_DIR     = os.path.join(DRIVE_DIR, "output")

# Use LOCAL SSD for fast I/O (lost on session restart, but features are cached to Drive)
GIF_DIR        = "/content/local_data/gifs"
FRAME_DIR      = "/content/local_data/frames"
FEATURE_DIR    = os.path.join(DRIVE_DIR, "features")  # Features persist on Drive

# Model
ENCODER_DIM    = 1280      # EfficientNet-B0 output
HIDDEN_SIZE    = 512       # BiLSTM hidden
EMBED_DIM      = 300       # GloVe
NUM_FRAMES     = 16        # Frames per GIF
MAX_SEQ_LEN    = 20        # Max caption length
VOCAB_LIMIT    = 6000      # Top-K words

# Training
NUM_EPOCHS     = 10
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-5
VAL_SPLIT      = 0.05

# GloVe
GLOVE_PATH     = os.path.join(DRIVE_DIR, "glove.6B.300d.txt")


# ============================================================
# STEP 1: Parse TGIF dataset
# ============================================================

def parse_tgif(tsv_path, max_samples=None):
    """Parse TGIF TSV: URL<tab>caption"""
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                url = parts[0].strip()
                caption = parts[1].strip().lower()
                gif_id = url.split('/')[-1].replace('.gif', '')
                data.append({'url': url, 'caption': caption, 'gif_id': gif_id})
                if max_samples and len(data) >= max_samples:
                    break
    return data


# ============================================================
# STEP 2: Download GIFs and extract frames
# ============================================================

def download_gif(url, save_path):
    """Download GIF if not already cached."""
    if os.path.exists(save_path):
        return True
    try:
        import urllib.request
        urllib.request.urlretrieve(url, save_path)
        return True
    except:
        return False


def extract_frames_pil(gif_path, num_frames=16):
    """Extract frames using PIL directly (much faster than ffmpeg)."""
    try:
        img = Image.open(gif_path)
        all_frames = []
        try:
            while True:
                all_frames.append(img.convert('RGB').copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if len(all_frames) == 0:
            return []

        # Select evenly spaced frames
        step = max(1, (len(all_frames) - 1) / (num_frames - 1)) if len(all_frames) > 1 else 1
        selected = []
        for i in range(num_frames):
            idx = min(int(i * step), len(all_frames) - 1)
            selected.append(all_frames[idx])

        # Pad if needed
        while len(selected) < num_frames:
            selected.append(selected[-1])

        return selected[:num_frames]
    except:
        return []


# ============================================================
# STEP 3: Extract EfficientNet features
# ============================================================

def extract_features_from_pil(pil_frames, cnn, transform, device, num_frames=16):
    """Extract EfficientNet features from PIL frames directly (no disk I/O)."""
    images = [transform(f) for f in pil_frames[:num_frames]]

    while len(images) < num_frames:
        images.append(images[-1] if images else torch.zeros(3, 224, 224))

    batch = torch.stack(images[:num_frames]).to(device)
    with torch.no_grad():
        features = cnn(batch)
    return features.cpu()


# ============================================================
# STEP 4: Build vocabulary
# ============================================================

def build_vocab(captions, vocab_limit=6000):
    """Build word2idx/idx2word from captions."""
    from collections import Counter
    word_counts = Counter()
    for cap in captions:
        words = cap.lower().split()
        word_counts.update(words)

    # Special tokens
    word2idx = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
    idx = 4
    for word, _ in word_counts.most_common(vocab_limit - 4):
        word2idx[word] = idx
        idx += 1

    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def load_glove(glove_path, word2idx, embed_dim=300):
    """Load GloVe embeddings for vocab."""
    embeddings = np.random.randn(len(word2idx), embed_dim).astype(np.float32) * 0.01
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"  GloVe: {found}/{len(word2idx)} words found")
    return embeddings


# ============================================================
# STEP 5: Model definition
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
        self.embedding.weight.requires_grad = False  # Freeze GloVe

        self.encoder_lstm = nn.LSTM(encoder_dim, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(embed_dim + hidden_size * 2, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_embeddings)

    def forward(self, features, captions, word2idx):
        """
        features: (batch, 16, 1280)
        captions: (batch, max_seq_len) - token indices
        """
        batch_size = features.size(0)
        device = features.device

        # Encode
        encoder_outputs, (h, c) = self.encoder_lstm(features)
        h = h.sum(dim=0).unsqueeze(0)  # (1, batch, hidden)
        c = c.sum(dim=0).unsqueeze(0)

        # Decode (teacher forcing)
        max_len = captions.size(1)
        outputs = torch.zeros(batch_size, max_len, len(word2idx)).to(device)

        for t in range(max_len):
            if t == 0:
                token = torch.full((batch_size,), word2idx['<start>'], dtype=torch.long).to(device)
            else:
                token = captions[:, t - 1]

            embed = self.embedding(token)  # (batch, embed_dim)
            attn_weights = self.attention(encoder_outputs, h[-1])
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            lstm_input = torch.cat((embed, context), dim=1).unsqueeze(1)
            output, (h, c) = self.decoder_lstm(lstm_input, (h, c))
            outputs[:, t] = self.fc(output.squeeze(1))

        return outputs


# ============================================================
# STEP 6: Dataset
# ============================================================

class TGIFDataset(Dataset):
    def __init__(self, features_list, captions_list, word2idx, max_seq_len=20):
        self.features = features_list
        self.captions = captions_list
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]  # (16, 1280)

        # Tokenize caption
        words = self.captions[idx].lower().split()
        tokens = [self.word2idx.get(w, self.word2idx['<unk>']) for w in words]
        tokens.append(self.word2idx['<end>'])
        tokens = tokens[:self.max_seq_len]

        # Pad
        padded = tokens + [self.word2idx['<pad>']] * (self.max_seq_len - len(tokens))

        return {
            'features': features,
            'captions': torch.tensor(padded, dtype=torch.long),
        }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CNN-LSTM TRAINING ON FULL TGIF DATASET")
    print("EfficientNet-B0 + BiLSTM + Attention")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)

    # ── Step 1: Parse TGIF and use official train/test splits ──
    print(f"\n1. Parsing TGIF dataset...")
    all_data = parse_tgif(TGIF_TSV)
    print(f"   {len(all_data):,} total samples loaded")

    # Download official TGIF splits from GitHub
    TRAIN_SPLIT = os.path.join(DRIVE_DIR, "train.txt")
    TEST_SPLIT  = os.path.join(DRIVE_DIR, "test.txt")

    if not os.path.exists(TRAIN_SPLIT) or not os.path.exists(TEST_SPLIT):
        print("   Downloading official TGIF splits...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/train.txt",
            TRAIN_SPLIT)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/splits/test.txt",
            TEST_SPLIT)

    with open(TRAIN_SPLIT) as f:
        train_urls = set(line.strip() for line in f)
    with open(TEST_SPLIT) as f:
        test_urls = set(line.strip() for line in f)
    print(f"   Official splits: {len(train_urls):,} train, {len(test_urls):,} test")

    # Filter to train split only
    data = [d for d in all_data if d['url'] in train_urls]
    print(f"   Training pool: {len(data):,} samples (test excluded)")

    # ── Step 2: Download GIFs + extract frames + features ──
    print(f"\n2. Processing GIFs (download → frames → features)...")
    print(f"   This will take several hours on first run. Features are cached.")

    cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    cnn.classifier = nn.Identity()
    cnn.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    valid_data = []
    processed = 0
    skipped = 0
    cached = 0
    t_start = time.time()

    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)

    # First pass: collect cached features (fast) + validate
    print("   Scanning cached features (filtering bad GIFs)...")
    to_process = []
    bad_cached = 0
    for item in data:
        gif_id = item['gif_id']
        feature_path = os.path.join(FEATURE_DIR, f"{gif_id}.pt")
        if os.path.exists(feature_path):
            try:
                feat = torch.load(feature_path, map_location='cpu', weights_only=True)
                if feat.shape != (NUM_FRAMES, ENCODER_DIM):
                    bad_cached += 1
                    continue
                # Check for near-zero features (placeholder GIF → blank features)
                if feat.abs().mean() < 0.01:
                    bad_cached += 1
                    continue
                # Check if all frames have identical features (static error image)
                if (feat[0] - feat[-1]).abs().max() < 0.001:
                    bad_cached += 1
                    continue
                valid_data.append({**item, 'feature_path': feature_path})
                cached += 1
            except:
                bad_cached += 1
        else:
            to_process.append(item)
    print(f"   Cached: {cached:,} | Bad/filtered: {bad_cached:,} | To process: {len(to_process):,}")

    # Second pass: parallel download + sequential GPU feature extraction
    DOWNLOAD_WORKERS = 16
    BATCH_DOWNLOAD = 64  # download 64 GIFs in parallel, then process on GPU

    def download_and_extract_frames(item):
        """Download GIF, validate, and extract PIL frames (runs in thread pool)."""
        gif_id = item['gif_id']
        gif_path = os.path.join(GIF_DIR, f"{gif_id}.gif")
        try:
            if not os.path.exists(gif_path):
                import urllib.request
                urllib.request.urlretrieve(item['url'], gif_path)

            # Filter broken/placeholder GIFs
            file_size = os.path.getsize(gif_path)
            if file_size < 5000:  # < 5KB = likely placeholder/error GIF
                os.remove(gif_path)
                return None

            frames = extract_frames_pil(gif_path, NUM_FRAMES)
            try:
                os.remove(gif_path)
            except:
                pass

            if len(frames) < 2:  # Error GIFs usually have 1 frame
                return None

            # Check for low-variance (solid color / "content not available" placeholder)
            import numpy as np
            sample = np.array(frames[0].resize((64, 64)))
            if sample.std() < 15:  # Very low variance = placeholder image
                return None

            return (item, frames)
        except:
            pass
        return None

    for batch_start in range(0, len(to_process), BATCH_DOWNLOAD):
        batch = to_process[batch_start:batch_start + BATCH_DOWNLOAD]

        # Parallel download + frame extraction
        results = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = {executor.submit(download_and_extract_frames, item): item for item in batch}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    skipped += 1

        # Sequential GPU feature extraction (fast)
        for item, pil_frames in results:
            gif_id = item['gif_id']
            feature_path = os.path.join(FEATURE_DIR, f"{gif_id}.pt")
            features = extract_features_from_pil(pil_frames, cnn, transform, device, NUM_FRAMES)
            torch.save(features, feature_path)
            valid_data.append({**item, 'feature_path': feature_path})
            processed += 1

        if processed % 500 == 0 or batch_start + BATCH_DOWNLOAD >= len(to_process):
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(to_process) - processed - skipped) / rate if rate > 0 else 0
            print(f"   Processed: {processed:,} | Cached: {cached:,} | Skipped: {skipped:,} | "
                  f"Rate: {rate:.1f}/s | ETA: {timedelta(seconds=int(remaining))}")

    print(f"   Total valid: {len(valid_data):,} (processed: {processed:,}, cached: {cached:,}, skipped: {skipped:,})")

    del cnn
    torch.cuda.empty_cache()

    # ── Step 3: Build vocabulary ──
    print(f"\n3. Building vocabulary...")
    captions = [d['caption'] for d in valid_data]
    word2idx, idx2word = build_vocab(captions, VOCAB_LIMIT)
    print(f"   Vocab size: {len(word2idx)}")

    # Save vocab
    with open(os.path.join(OUTPUT_DIR, "word2idx.json"), 'w') as f:
        json.dump(word2idx, f)
    with open(os.path.join(OUTPUT_DIR, "idx2word.json"), 'w') as f:
        json.dump({str(k): v for k, v in idx2word.items()}, f)

    # ── Step 4: Load GloVe embeddings ──
    print(f"\n4. Loading GloVe embeddings...")
    if os.path.exists(GLOVE_PATH):
        embed_matrix = load_glove(GLOVE_PATH, word2idx, EMBED_DIM)
    else:
        print(f"   GloVe not found at {GLOVE_PATH}, using random init")
        embed_matrix = np.random.randn(len(word2idx), EMBED_DIM).astype(np.float32) * 0.01

    np.save(os.path.join(OUTPUT_DIR, "embedding_matrix.npy"), embed_matrix)

    # ── Step 5: Create datasets ──
    print(f"\n5. Loading features and creating datasets...")
    all_features = []
    all_captions = []
    for item in valid_data:
        try:
            feat = torch.load(item['feature_path'], map_location='cpu', weights_only=True)
            if feat.shape == (NUM_FRAMES, ENCODER_DIM):
                all_features.append(feat)
                all_captions.append(item['caption'])
        except:
            pass

    print(f"   Loaded {len(all_features):,} feature sets")

    val_size = int(len(all_features) * VAL_SPLIT)
    train_features = all_features[val_size:]
    train_captions = all_captions[val_size:]
    val_features = all_features[:val_size]
    val_captions = all_captions[:val_size]

    train_dataset = TGIFDataset(train_features, train_captions, word2idx, MAX_SEQ_LEN)
    val_dataset = TGIFDataset(val_features, val_captions, word2idx, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"   Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # ── Step 6: Create model ──
    print(f"\n6. Creating model...")
    model = CaptionGenerator(embed_matrix, ENCODER_DIM, HIDDEN_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable: {trainable:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ── Step 7: Train ──
    print(f"\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            captions = batch['captions'].to(device)

            outputs = model(features, captions, word2idx)

            # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len)
            loss = criterion(outputs.view(-1, len(word2idx)), captions.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                captions = batch['captions'].to(device)
                outputs = model(features, captions, word2idx)
                loss = criterion(outputs.view(-1, len(word2idx)), captions.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Time: {timedelta(seconds=int(elapsed))}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "caption_bilstm_attention_best.pt"))
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt"))

    # ── Step 8: Save final model ──
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "caption_bilstm_attention.pt"))

    # Save training log
    log = {
        "config": {
            "encoder": "EfficientNet-B0",
            "decoder": "BiLSTM + Attention",
            "encoder_dim": ENCODER_DIM,
            "hidden_size": HIDDEN_SIZE,
            "embed_dim": EMBED_DIM,
            "vocab_size": len(word2idx),
            "num_frames": NUM_FRAMES,
            "max_seq_len": MAX_SEQ_LEN,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "total_params": total_params,
            "trainable_params": trainable,
        },
        "data": {
            "dataset": "TGIF",
            "total_samples": len(valid_data),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "training_time_seconds": total_time,
        "training_time_formatted": str(timedelta(seconds=int(total_time))),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }

    with open(os.path.join(OUTPUT_DIR, "training_log.json"), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"Training time: {log['training_time_formatted']}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"Files: caption_bilstm_attention.pt, word2idx.json, idx2word.json, embedding_matrix.npy")
    print(f"\nNext: Export to ONNX and update the Chrome extension")


if __name__ == "__main__":
    main()
