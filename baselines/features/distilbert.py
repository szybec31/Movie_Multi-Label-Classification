# features/distilbert.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import os

class DistilBERTEmbedder:
    def __init__(self, model_name="distilbert-base-uncased", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, batch_size=16, max_length=256):
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state.mean(dim=1)

            embeddings.append(cls_emb.cpu().numpy())

        return np.vstack(embeddings)

def build_distilbert_embedding(X,split):
    EMB_PATH = "cache/bert_embeddings.npy"     #"bert_embeddings.npy"


    # sprawdzenie czy plik istnieje
    if os.path.exists(EMB_PATH):
        print("Wczytywanie embeddingów z pliku...")
        X_bert = np.load(EMB_PATH)

    else:

        print("Generowanie embeddingów BERT...")
        embedder = DistilBERTEmbedder()

        X_bert = embedder.encode(
            X,
            batch_size=32
        )

        print("Zapisywanie embeddingów...")
        np.save(EMB_PATH, X_bert)

    print("DistilBERTEmbedder Ilość cech: ",X_bert.shape[1])
    train_idx, test_idx = split
    X_train, X_test = X_bert[train_idx], X_bert[test_idx]

    return X_train,X_test