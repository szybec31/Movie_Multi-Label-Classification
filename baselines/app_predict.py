import os
import joblib
import numpy as np
import torch
from baselines.features.resnet50 import extract_single as resnet50_extract_single
from baselines.features.distilbert import extract_single as distilbert_extract_single, DistilBERTEmbedder

FROZEN_DIR = "baselines/frozen_models"
MODEL_CONFIGS = {
    "text": os.path.join(FROZEN_DIR, "text_tfidf_logistic.pkl"),
    "graphics": os.path.join(FROZEN_DIR, "graphics_resnet50_logistic.pkl"),
    "early-fusion": os.path.join(FROZEN_DIR, "early-fusion_tfidf_resnet50_logistic.pkl")
}


class MoviePredictor:
    # prediction_mode: "text", "graphics" lub "early-fusion"
    def __init__(self, prediction_mode: str):
        if prediction_mode not in MODEL_CONFIGS:
            raise ValueError(f"Unknown predict type \"{prediction_mode}\"")

        model_path = MODEL_CONFIGS[prediction_mode]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find model called \"{model_path}\". Run main.py with test_type=\"freeze\" first")

        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.vectorizers = self.bundle["vectorizers"]
        self.mlb = self.bundle["mlb"]
        self.threshold = self.bundle["threshold"]
        self.config = self.bundle["config"]

        self.bert_embedder = None

    def _get_text_features(self, title: str, overview: str):
        full_text = f"{str(title)} {str(overview)}"
        vec_name = self.config["vectorizers"][0]
        text_vectorizer = self.vectorizers[0]

        if vec_name == "tfidf":
            if text_vectorizer is None:
                raise ValueError("Error: there is no TfidfVectorizer in the model file.")
            feat = text_vectorizer.transform([full_text])
            return feat.toarray()

        elif vec_name == "distilbert":
            if self.bert_embedder is None:
                self.bert_embedder = DistilBERTEmbedder()

            feat = distilbert_extract_single(full_text, embedder=self.bert_embedder)
            return feat.reshape(1, -1)  # Zmiana wymiaru na tablicę 2D (1, 768)

    def _get_image_features(self, image_path: str):
        vec_idx = 0 if self.config["type"] == "graphics" else 1
        vec_name = self.config["vectorizers"][vec_idx]

        if vec_name == "resnet50":
            feat = resnet50_extract_single(image_path)
            return feat.reshape(1, -1)  # Zmiana wymiaru na tablicę 2D (1, 2048)
        else:
            raise ValueError(f"Unsupported image vectorizer: {vec_name}")

    def predict(self, title: str = "", overview: str = "", image_path: str = None):
        features = []

        if self.config["type"] in ["text", "early-fusion"]:
            text_feat = self._get_text_features(title, overview)
            features.append(text_feat)

        if self.config["type"] in ["graphics", "early-fusion"]:
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"There is no such image under the path: {image_path}")
            img_feat = self._get_image_features(image_path)
            features.append(img_feat)

        # 3. Early-Fusion (połączenie wektorów w jeden szeroki)
        if self.config["type"] == "early-fusion":
            X_input = np.hstack(features)
        else:
            X_input = features[0]

        # 4. Predykcja prawdopodobieństw klas
        probabilities = self.model.predict_proba(X_input)

        # Normalizacja wyników wyjściowych dla MultiOutputClassifier
        if isinstance(probabilities, list):
            prob_matrix = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probabilities]).T
        else:
            prob_matrix = probabilities

        # 5. Zastosowanie thresholdów do wektora binarnego (0 lub 1)
        binary_prediction = (prob_matrix >= self.threshold).astype(int)

        # 6. Dekodowanie za pomocą odzyskanego z pliku pkl MultiLabelBinarizer (LabelTransform.mlb)
        if self.mlb is not None:
            predicted_genres = self.mlb.inverse_transform(binary_prediction)
            return list(predicted_genres[0])

        return binary_prediction.tolist()
