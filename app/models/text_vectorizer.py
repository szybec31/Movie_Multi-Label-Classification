# app/models/text_vectorizer.py

import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel
)


class TextVectorizer:

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        max_length=256,
        device=None
    ):

        self.max_length = max_length

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        try:
            # =========================
            # LOAD TOKENIZER
            # =========================
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False
            )

            # =========================
            # LOAD MODEL
            # =========================
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=False
            )

            self.model.eval()
            self.model.to(self.device)

            self.ready = True

        except Exception as e:

            print("ERROR: Cannot load DistilBERT model")
            print("Reason:", e)

            # fallback flag
            self.ready = False
            self.tokenizer = None
            self.model = None

    # =========================================
    # SINGLE TEXT
    # =========================================
    def encode(self, text):
        if not self.ready:
            # fallback bez modelu
            return np.zeros(768, dtype=np.float32)

        if not text:
            return np.zeros(768, dtype=np.float32)

        # tokenizer
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # inference
        with torch.no_grad():

            outputs = self.model(**inputs)

            # mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)

        # tensor -> numpy
        embedding = (
            embedding
            .cpu()
            .numpy()
            .flatten()
            .astype(np.float32)
        )

        return embedding

    # =========================================
    # MULTI TEXTS (OPTIONAL)
    # =========================================
    def encode_batch(
        self,
        texts,
        batch_size=16
    ):

        embeddings = []

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                outputs = self.model(**inputs)

                batch_embeddings = (
                    outputs
                    .last_hidden_state
                    .mean(dim=1)
                )

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

        return np.vstack(embeddings)