import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 nowy sposób (bez warningów)
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

transform = weights.transforms()


# ========================
# SINGLE IMAGE
# ========================
def extract_single(path):
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(img).cpu().numpy().flatten()
    except:
        feat = np.zeros(512)

    return feat


# ========================
# MAIN CACHE FUNCTION
# ========================
def load_or_compute_features(paths, cache_path="cache/image_features.npy"):

    os.makedirs("cache", exist_ok=True)

    # 🔥 jeśli cache istnieje → wczytaj
    if os.path.exists(cache_path):
        print("Loading cached image features...")
        return np.load(cache_path)

    print("Computing image features...")

    features = []
    for p in tqdm(paths):
        features.append(extract_single(p))

    features = np.array(features)

    np.save(cache_path, features)
    print(f"Saved cache to {cache_path}")

    return features


# ========================
# BUILD FOR PIPELINE
# ========================
def build_image_features(df, split_idx=None):

    feats = load_or_compute_features(df["poster_path"])

    if split_idx is None:
        return feats, None, None

    train_idx, test_idx = split_idx

    return feats[train_idx], feats[test_idx], None