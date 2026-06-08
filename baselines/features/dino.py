import os
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# LOAD DINOv2
# ========================

MODEL_NAME = "facebook/dinov2-base"


processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval().to(device)


# ========================
# SINGLE IMAGE
# ========================

def extract_single(path):
    try:
        image = Image.open(path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            # CLS token embedding
            feat = outputs.last_hidden_state[:, 0]

        feat = feat.cpu().numpy().flatten()

    except Exception:
        # DINOv2 base = 768 dims
        feat = np.zeros(768)

    return feat


# ========================
# CACHE
# ========================

def load_or_compute_features(
    paths,
    cache_path="cache/dinov2.npy"
):

    os.makedirs("cache", exist_ok=True)

    if os.path.exists(cache_path):
        print("Loading cached DINOv2 features...")
        return np.load(cache_path)

    print("Computing DINOv2 features...")

    features = []

    for p in tqdm(paths):
        features.append(extract_single(p))

    features = np.array(features)

    np.save(cache_path, features)

    print(f"Saved cache to {cache_path}")

    return features


# ========================
# PIPELINE ENTRY
# ========================

def build_image_features(df, split_idx=None):

    feats = load_or_compute_features(df["poster_path"])

    print("DINOv2 Ilość cech:", feats.shape[1])

    if split_idx is None:
        return feats, None, None

    train_idx, test_idx = split_idx

    return (
        feats[train_idx],
        feats[test_idx],
        None
    )