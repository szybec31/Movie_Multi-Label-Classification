import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

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
        feat = np.zeros(2048)

    return feat


# ========================
# MAIN CACHE FUNCTION
# ========================
def load_or_compute_features(paths, cache_path="cache/resnet50.npy"):

    os.makedirs("cache", exist_ok=True)

    # jeśli cache istnieje → wczytaj
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
    print("ResNet50 Ilość cech: ", feats.shape[1])

    if split_idx is None:
        return feats, None, None

    train_idx, test_idx = split_idx

    return feats[train_idx], feats[test_idx], None