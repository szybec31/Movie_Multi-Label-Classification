import os
import joblib
import numpy as np
from .utils.remake_config import clean_model_config
from .models.logistic import train_logistic
from .models.svm import train_svm
from .models.randomforest import train_random_forest
from .models.mlp import train_mlp
from .features.tfidf import build_tfidf
from .features.distilbert import build_distilbert_embedding
from .features.resnet50 import build_image_features as build_resnet50
from .features.resnet18 import build_image_features as build_resnet18

SAVE_DIR = "frozen_models"

def ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def train_single_model(X, y, model_name, config, idx=0):
    balanced = config.get("balanced", False)
    if "balanced_list" in config:
        balanced = config["balanced_list"][idx]

    if model_name == "logistic":
        model = train_logistic(X, y, balanced)

    elif model_name == "svm":
        model = train_svm(X, y, balanced, **clean_model_config(config, ["balanced"]))

    elif model_name == "random_forest":
        model = train_random_forest(X, y, balanced=balanced, **clean_model_config(config, ["balanced"]))

    elif model_name == "mlp":
        model = train_mlp(X, y, **config)

    else:
        raise ValueError("Unknown model")

    return model


def build_features(df, config):
    features = []
    vectorizers = []
    if config["type"] in ["text", "early-fusion", "late-fusion"]:
        text_data = (df["title"].fillna("") + " " + df["overview"].fillna(""))
        vec = config["vectorizers"][0]

        if vec == "tfidf":
            X, _, vectorizer = build_tfidf(
                text_data,
                text_data,
                fit_full=True,
                **config
            )
            features.append(X)
            vectorizers.append(vectorizer)

        elif vec == "distilbert":
            X, _ = build_distilbert_embedding(
                text_data,
                split=None
            )
            features.append(X)
            vectorizers.append(None)

    if config["type"] in ["graphics", "early-fusion", "late-fusion"]:
        vec_idx = 0 if config["type"] == "graphics" else 1
        vec = config["vectorizers"][vec_idx]

        if vec == "resnet50":
            X, _, _ = build_resnet50(
                df,
                split_idx=None
            )

        elif vec == "resnet18":
            X, _, _ = build_resnet18(
                df,
                split_idx=None
            )

        features.append(X)
        vectorizers.append(None)

    return features, vectorizers


def get_base_filename(config):
    model_names = "_".join(config["models"])
    vectorizer_names = "_".join(config["vectorizers"])

    return (
        f"{config['type']}_"
        f"{vectorizer_names}_"
        f"{model_names}"
    )


def freeze_model(df, y, config, mlb=None):
    print("FREEZING MODEL")
    print(config)

    features, vectorizers = build_features(df, config)
    ensure_dir()

    base_name = get_base_filename(config)

    if config["type"] in ["text", "graphics"]:
        X = features[0]
        model = train_single_model(
            X,
            y,
            config["models"][0],
            config
        )

        save_data = {
            "model": model,
            "vectorizers": vectorizers,
            "mlb": mlb,
            "config": config,
            "threshold": config.get("threshold", 0.5)
        }

        joblib.dump(
            save_data,
            os.path.join(SAVE_DIR, f"{base_name}.pkl")
        )

    elif config["type"] == "early-fusion":
        X = np.hstack(features)
        model = train_single_model(
            X,
            y,
            config["models"][0],
            config
        )

        save_data = {
            "model": model,
            "vectorizers": vectorizers,
            "mlb": mlb,
            "config": config,
            "threshold": config.get("threshold", 0.5)
        }

        joblib.dump(
            save_data,
            os.path.join(SAVE_DIR, f"{base_name}.pkl")
        )

    elif config["type"] == "late-fusion":
        models = []
        for i in range(2):
            model = train_single_model(
                features[i],
                y,
                config["models"][i],
                config,
                idx=i
            )
            models.append(model)

        save_data = {
            "models": models,
            "vectorizers": vectorizers,
            "mlb": mlb,
            "config": config,
            "thresholds": config.get("thresholds", [0.5, 0.5])
        }

        joblib.dump(
            save_data,
            os.path.join(SAVE_DIR, f"{base_name}.pkl")
        )

    print(f"\nSaved model bundle to: {SAVE_DIR}\n")
