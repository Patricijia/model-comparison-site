#!/usr/bin/env python3
"""
Generate captions for example GIFs using all 3 models:
1. CNN-LSTM Baseline (EfficientNet-B0 + BiLSTM + Attention)
2. ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)
3. SmolVLM-256M Distilled

Mimics the extension pipeline: download GIF → extract frames → run model on frames → summarize
"""

import os
import json
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
)
from peft import PeftModel
import urllib.request
import time

# ── GIFs to caption ──────────────────────────────────────────────────────────
GIFS = [
    {
        "url": "https://38.media.tumblr.com/788a5bbb7b0c4432b9d6699b4da327f2/tumblr_nhv8yzFmkd1u3l54uo1_400.gif",
        "id": "tumblr_nhv8yzFmkd1u3l54uo1_400",
        "baseline_caption": "a man is playing a guitar and a guitar is playing the drums",
    },
    {
        "url": "https://38.media.tumblr.com/fab4bf9cfc2470a6cbeb8133ed170354/tumblr_no4ptzwyDv1qm9n8co1_400.gif",
        "id": "tumblr_no4ptzwyDv1qm9n8co1_400",
        "baseline_caption": "a man with long hair is smoking a cigarette",
    },
    {
        "url": "https://38.media.tumblr.com/b10883172762e3f05d169d1146e6eac1/tumblr_nqrrpz9Ke81qhvqi6o1_400.gif",
        "id": "tumblr_nqrrpz9Ke81qhvqi6o1_400",
        "baseline_caption": "a man is smiling and moving his head",
    },
    {
        "url": "https://38.media.tumblr.com/d5163408cc6b787661a0eb3fa439b9e5/tumblr_ni4qhxft1T1tnkki4o1_400.gif",
        "id": "tumblr_ni4qhxft1T1tnkki4o1_400",
        "baseline_caption": "a young man is smiling and laughing",
    },
    {
        "url": "https://38.media.tumblr.com/43138cf3dbb61c6b0bb922b6ee346d76/tumblr_notz47rcwY1sq24x8o1_400.gif",
        "id": "tumblr_notz47rcwY1sq24x8o1_400",
        "baseline_caption": "a woman is walking through a door and a man is standing behind her",
    },
    {
        "url": "https://38.media.tumblr.com/9d1b882d1ac6d85c015d57939fa7e74c/tumblr_npdeeniOGl1s06j30o1_400.gif",
        "id": "tumblr_npdeeniOGl1s06j30o1_400",
        "baseline_caption": "a car is driving on a track",
    },
]

WORK_DIR = tempfile.mkdtemp(prefix="gif_captions_")


# ── Summarize captions (ported from Chrome extension offscreen.js) ───────────
def summarize_captions(captions):
    """
    Same algorithm as the browser extension's summarizeCaptions():
    1. Tokenize each caption, filter stop words
    2. Score by representativeness (avg word overlap) + informativeness (word count)
    3. Pick highest-scoring as primary
    4. Find distinctive detail from remaining captions
    5. Cap at 120 chars
    """
    if not captions:
        return ""
    if len(captions) == 1:
        return captions[0]

    import re

    STOP_WORDS = {
        "a", "an", "the", "is", "in", "of", "on", "at", "to", "and", "with",
        "for", "from", "that", "this", "are", "was", "has", "his", "her", "its",
    }

    def tokenize(text):
        return [w for w in re.sub(r"[^a-z ]", "", text.lower()).split() if len(w) > 1]

    def content_words(tokens):
        return [w for w in tokens if w not in STOP_WORDS]

    def word_overlap(set_a, set_b):
        if not set_a or not set_b:
            return 0
        shared = len(set_a & set_b)
        return shared / max(len(set_a), len(set_b))

    tokenized = []
    for c in captions:
        tokens = tokenize(c)
        cset = set(content_words(tokens))
        tokenized.append({"original": c, "tokens": tokens, "content_set": cset})

    # Score each caption
    scored = []
    max_content = max(len(t["content_set"]) for t in tokenized) or 1
    for i, item in enumerate(tokenized):
        total_overlap = 0
        comparisons = 0
        for j, other in enumerate(tokenized):
            if j == i:
                continue
            total_overlap += word_overlap(item["content_set"], other["content_set"])
            comparisons += 1
        representativeness = total_overlap / comparisons if comparisons > 0 else 0
        informativeness = len(item["content_set"]) / max_content
        score = 0.7 * representativeness + 0.3 * informativeness
        scored.append({**item, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    primary = scored[0]

    # Find distinctive detail from remaining captions
    best_detail = None
    best_detail_score = -1

    for other in scored[1:]:
        new_words = other["content_set"] - primary["content_set"]
        if not new_words:
            continue
        # Split on common conjunctions to get clauses
        clauses = re.split(r"\b(?:is |with |and |in )\b", other["original"].lower())
        clauses = [c.strip() for c in clauses if len(c.strip()) > 2]

        for clause in clauses:
            clause_words = set(content_words(tokenize(clause)))
            new_count = len(clause_words - primary["content_set"])
            if new_count > best_detail_score and clause_words:
                best_detail_score = new_count
                detail = re.sub(r"^(a |an |the |is |are )", "", clause.strip()).strip()
                best_detail = detail

    result = primary["original"]
    if best_detail and len(best_detail) > 2 and best_detail not in result.lower():
        result += " and " + best_detail

    if len(result) > 120:
        result = result[:117] + "..."

    return result
BASELINE_MODEL_PATH = "/home/patricija/Desktop/Thesis/GIFreader/GifModelRe/caption_bilstm_attention.pt"
BASELINE_EMBED_PATH = "/home/patricija/Desktop/Thesis/GIFreader/GifModelRe/embedding_matrix_300d.npy"
BASELINE_VOCAB_PATH = "/home/patricija/Desktop/Thesis/GIFreader/GifModelRe/word2idx.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Work dir: {WORK_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Download GIFs and extract frames
# ══════════════════════════════════════════════════════════════════════════════
def download_gif(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


def extract_frames(gif_path, out_dir, num_frames=16):
    """Extract frames using ffmpeg, like the extension does."""
    os.makedirs(out_dir, exist_ok=True)
    # Get total frames first
    cmd = [
        "ffmpeg", "-y", "-i", gif_path,
        "-vf", f"fps={num_frames}/1",
        os.path.join(out_dir, "frame_%03d.jpg"),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted(
        [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")]
    )
    return frames[:num_frames]


def extract_frames_pil(gif_path, num_frames=4):
    """Extract frames using PIL (like the browser extension does for ViT-GPT2)."""
    img = Image.open(gif_path)
    frames = []
    try:
        total = 0
        while True:
            img.seek(total)
            total += 1
    except EOFError:
        pass

    if total == 0:
        return []

    step = max(1, total // num_frames)
    for i in range(0, total, step):
        if len(frames) >= num_frames:
            break
        img.seek(i)
        frames.append(img.convert("RGB").copy())
    return frames


print("\n" + "=" * 70)
print("DOWNLOADING GIFS AND EXTRACTING FRAMES")
print("=" * 70)

for gif in GIFS:
    gif_path = os.path.join(WORK_DIR, f"{gif['id']}.gif")
    download_gif(gif["url"], gif_path)
    gif["local_path"] = gif_path

    # Extract 16 frames for baseline (ffmpeg)
    frame_dir = os.path.join(WORK_DIR, f"{gif['id']}_frames")
    gif["frames_16"] = extract_frames(gif_path, frame_dir, 16)

    # Extract 4 frames for ViT-GPT2 (PIL, like browser extension)
    gif["frames_4"] = extract_frames_pil(gif_path, 4)

    print(f"  {gif['id']}: {len(gif['frames_16'])} frames (baseline), {len(gif['frames_4'])} frames (vit-gpt2)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: CNN-LSTM Baseline model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 1: CNN-LSTM BASELINE")
print("=" * 70)


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, hidden), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class CaptionGenerator(nn.Module):
    def __init__(self, embed_matrix, encoder_dim=1280, hidden_size=512):
        super().__init__()
        num_embeddings, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.embedding.weight.requires_grad = False
        self.encoder_lstm = nn.LSTM(
            encoder_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.decoder_lstm = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, batch_first=True
        )
        self.attention = Attention(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_embeddings)


def run_baseline(gifs):
    with open(BASELINE_VOCAB_PATH, "r") as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}

    embed_matrix = np.load(BASELINE_EMBED_PATH)
    baseline_model = CaptionGenerator(embed_matrix).to(device)
    baseline_model.load_state_dict(
        torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=True)
    )
    baseline_model.eval()

    # EfficientNet-B0 for feature extraction
    cnn = efficientnet_b0(weights="IMAGENET1K_V1")
    cnn.classifier = nn.Identity()
    cnn.eval().to(device)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    for gif in gifs:
        t0 = time.time()
        frames = gif["frames_16"]
        if len(frames) < 16:
            # Pad by repeating last frame
            while len(frames) < 16:
                frames.append(frames[-1])

        images = torch.stack([transform(Image.open(p)) for p in frames[:16]]).to(device)
        with torch.no_grad():
            features = cnn(images)  # (16, 1280)

        # Generate caption
        features_in = features.unsqueeze(0)
        encoder_outputs, (h, c) = baseline_model.encoder_lstm(features_in)
        h = h.sum(dim=0).unsqueeze(0)
        c = c.sum(dim=0).unsqueeze(0)

        caption_words = []
        prev_word = "<start>"
        for _ in range(20):
            token = word2idx.get(prev_word, word2idx["<unk>"])
            embed = baseline_model.embedding(
                torch.tensor([[token]]).to(device)
            ).squeeze(1)
            attn_weights = baseline_model.attention(encoder_outputs, h[-1])
            context = torch.bmm(
                attn_weights.unsqueeze(1), encoder_outputs
            ).squeeze(1)
            lstm_input = torch.cat((embed, context), dim=1).unsqueeze(1)
            output, (h, c) = baseline_model.decoder_lstm(lstm_input, (h, c))
            scores = baseline_model.fc(output.squeeze(1))
            next_id = scores.argmax(dim=1).item()
            next_word = idx2word.get(next_id, "<unk>")
            if next_word == "<end>":
                break
            caption_words.append(next_word)
            prev_word = next_word

        elapsed = (time.time() - t0) * 1000
        caption = " ".join(caption_words)
        gif["baseline_generated"] = caption
        gif["baseline_ms"] = round(elapsed)
        print(f"  [{elapsed:.0f}ms] {gif['id']}: {caption}")

    del baseline_model, cnn
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


run_baseline(GIFS)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: ViT-GPT2 (like the browser extension, but in Python)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 2: VIT-GPT2")
print("=" * 70)


def run_vitgpt2(gifs):
    print("  Loading vit-gpt2-image-captioning...")
    vit_model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    ).to(device)
    vit_processor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    vit_tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    vit_model.eval()
    print("  Model loaded.")

    for gif in gifs:
        t0 = time.time()
        frames = gif["frames_4"]
        if not frames:
            gif["vitgpt2_caption"] = "No frames extracted"
            gif["vitgpt2_ms"] = 0
            continue

        # Caption each frame (like the extension does)
        per_frame_captions = []
        for frame in frames:
            pixel_values = vit_processor(images=frame, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                output_ids = vit_model.generate(
                    pixel_values, max_length=16, num_beams=4
                )
            caption = vit_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            per_frame_captions.append(caption)

        # Summarize using the same algorithm as the Chrome extension (offscreen.js)
        final_caption = summarize_captions(per_frame_captions)

        elapsed = (time.time() - t0) * 1000
        gif["vitgpt2_caption"] = final_caption
        gif["vitgpt2_ms"] = round(elapsed)
        gif["vitgpt2_per_frame"] = per_frame_captions
        print(f"  [{elapsed:.0f}ms] {gif['id']}: {final_caption}")
        print(f"           per-frame: {per_frame_captions}")

    del vit_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


run_vitgpt2(GIFS)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: SmolVLM-256M Distilled
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MODEL 3: SMOLVLM-256M DISTILLED")
print("=" * 70)


def run_smolvlm(gifs):
    try:
        print("  Loading SmolVLM-256M-Instruct base model...")
        base_model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
        processor = AutoProcessor.from_pretrained(base_model_id)
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_id, torch_dtype=torch.float32
        ).to(device)

        # Load LoRA adapter from HuggingFace Hub
        print("  Loading LoRA adapter from Patricijia/smolvlm-gif-descriptor...")
        smol_model = PeftModel.from_pretrained(base_model, "Patricijia/smolvlm-gif-descriptor")

        smol_model.eval()

        prompt = (
            "These frames are ordered left to right over time. "
            "Describe these frames in a short, simple sentence (max 10 words), "
            "similar to: 'a man walks across the room'. "
            "Use plain language and do not add extra commentary."
        )

        for gif in gifs:
            t0 = time.time()
            frames = gif["frames_4"]
            if not frames:
                gif["smolvlm_caption"] = "No frames extracted"
                gif["smolvlm_ms"] = 0
                continue

            # Create a grid image from frames (like SmolVLM expects)
            # Use 4 frames side by side
            widths = [f.width for f in frames]
            heights = [f.height for f in frames]
            total_w = sum(widths)
            max_h = max(heights)
            grid = Image.new("RGB", (total_w, max_h))
            x_offset = 0
            for f in frames:
                grid.paste(f, (x_offset, 0))
                x_offset += f.width

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=text_prompt, images=[grid], return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output_ids = smol_model.generate(**inputs, max_new_tokens=30)

            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            # Extract only the assistant response
            if "Assistant:" in caption:
                caption = caption.split("Assistant:")[-1].strip()
            elif "assistant" in caption.lower():
                parts = caption.lower().split("assistant")
                caption = parts[-1].strip().lstrip(":").strip()

            elapsed = (time.time() - t0) * 1000
            gif["smolvlm_caption"] = caption
            gif["smolvlm_ms"] = round(elapsed)
            print(f"  [{elapsed:.0f}ms] {gif['id']}: {caption}")

        del smol_model, base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"  SmolVLM error: {e}")
        for gif in gifs:
            if "smolvlm_caption" not in gif:
                gif["smolvlm_caption"] = f"Error: {e}"
                gif["smolvlm_ms"] = 0


run_smolvlm(GIFS)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Save results and print summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results = []
for gif in GIFS:
    r = {
        "id": gif["id"],
        "url": gif["url"],
        "baseline_caption": gif.get("baseline_generated", gif["baseline_caption"]),
        "vitgpt2_caption": gif.get("vitgpt2_caption", ""),
        "smolvlm_caption": gif.get("smolvlm_caption", ""),
        "baseline_ms": gif.get("baseline_ms", 0),
        "vitgpt2_ms": gif.get("vitgpt2_ms", 0),
        "smolvlm_ms": gif.get("smolvlm_ms", 0),
    }
    results.append(r)
    print(f"\n  GIF: {gif['id']}")
    print(f"    Baseline:  {r['baseline_caption']}")
    print(f"    ViT-GPT2:  {r['vitgpt2_caption']}")
    print(f"    SmolVLM:   {r['smolvlm_caption']}")

out_path = os.path.join(
    os.path.dirname(__file__), "caption_results.json"
)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
