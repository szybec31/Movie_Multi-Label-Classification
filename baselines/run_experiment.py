import numpy as np
from sklearn.metrics import f1_score
from .utils.remake_config import clean_model_config
from .utils.metrics import evaluate
from .utils.clear import clean_text


def run_experiment(df, y, split=None, **config):
    # ========================
    # type: str = "text" or "graphics" or "early-fusion" or "late-fusion"
    # subtype: str = "text" or "title" or "overview" (only for type = "text")
    # vectorizers: list[str] = up to 2 vectorizers from list above
    # models: list[str] = up to 2 models from list above
    # balanced: bool = True or False
    # balanced_list: list[bool] = list of using balanced params in models (only for "late-fusion")
    # threshold: float = 0.2, 0.3, 0.5 (only for tfidf vectorizer), base value = 0.5 for tfidf or None (for other vect)
    # thresholds: list[float] = list of thresholds for late-fusion when using min one model based on tfidf vectorizer
    # max_features_tfidf: int = base 20000,
    # ngram_range_tfidf: tuple = base (1,2),
    # n_estimators: int = base 200, for random_forest
    # max_depth: int = base 20, for random_forest
    # max_features_rf: str = base 'sqrt', for random_forest
    # hidden_layer_sizes: tuple = base (256, 128), for mlp
    # max_iter: int = base 20, for mlp
    # batch_size: int = base 64, for mlp
    # learning_rate_init: float = base 0.001, for mlp
    # ========================
    ## OLD:
    ## model: str = "logistic" or "svm" or "random_forest" or "mlp"
    ## vectorizer: str = "tfidf" or "distilbert" (for "text") or "resnet18" or "resnet50" (for "graphics")

    # ========================
    # VALIDATE all data
    # ========================
    if "type" not in config:
        raise ValueError("Unknown type")

    if config["type"] in ["text", "graphics"]:
        if "models" not in config or "vectorizers" not in config:
            raise ValueError("Set model and vectorizer param")

    elif config["type"] in ["early-fusion", "late-fusion"]:
        if "vectorizers" not in config:
            raise ValueError("Set vectorizers[text, graphics] param")
        if "models" not in config and config["type"] == "late-fusion":
            raise ValueError("Set models[text, graphics] param")
        if "model" not in config and "models" not in config and config["type"] == "early-fusion":
            raise ValueError("Set model or models[] param")

    if "models" not in config:
        config["models"] = [config["model"]]

    if "vectorizers" not in config:
        config["vectorizers"] = [config["vectorizer"]]

    if "thresholds" not in config:
        config["thresholds"] = []
        if "threshold" in config:
            config["thresholds"] = [config["threshold"]]

    if "balanced_list" not in config:
        if "balanced" in config:
            config["balanced_list"] = [config["balanced"]]
        else:
            config["balanced_list"] = [False, False]

    ## LISTS:
    X_list = []

    if config["type"] in ["text", "early-fusion", "late-fusion"]:
        if config["vectorizers"][0] not in ["tfidf", "distilbert"]:
            raise ValueError("Wrong vectorizer to chosen type; choose tfidf or distilbert")

        # ========================
        # SUBTYPE (for text only)
        # ========================
        if "subtype" not in config:
            config["subtype"] = "text"

        if config["subtype"] == "text":
            df["text"] = df["title"].fillna('') + " " + df["overview"].fillna('')
            if config["vectorizers"][0] == "tfidf":
                X1 = df["text"].apply(clean_text)
            else:
                X1 = df["text"]
        elif config["subtype"] in ["title", "overview"]:
            X1 = df[config["subtype"]]

        X_list.append(X1)

    if config["type"] == "graphics":
        if config["vectorizers"][0] not in ["resnet18", "resnet50", "clip", "dino"]:
            raise ValueError("Wrong vectorizer to chosen type; choose resnet18 or resnet50")
        X1 = df["poster_path"]
        X_list.append(X1)

    if config["type"] in ["early-fusion", "late-fusion"]:
        if config["vectorizers"][1] not in ["resnet18", "resnet50", "clip", "dino"]:
            raise ValueError("Wrong vectorizer to chosen type; choose resnet18 or resnet50")
        X2 = df["poster_path"]
        X_list.append(X2)

    # ========================
    # SPLIT
    # ========================
    if split is None:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42
        )
    else:
        train_idx, test_idx = split

    X_train_list = [X.iloc[train_idx] for X in X_list]
    X_test_list = [X.iloc[test_idx] for X in X_list]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # ========================
    # FEATURES / VECTORIZER
    # ========================
    features_train = []
    features_test = []

    for i, vec in enumerate(config["vectorizers"]):

        if vec == "tfidf":
            from .features.tfidf import build_tfidf
            Xt, Xv, _ = build_tfidf(X_train_list[i], X_test_list[i], **config)

        elif vec == "distilbert":
            from .features.distilbert import build_distilbert_embedding
            Xt, Xv, = build_distilbert_embedding(X_list[i], split=split)

        elif vec == "resnet50":
            from .features.resnet50 import build_image_features
            Xt, Xv, _ = build_image_features(df, (train_idx, test_idx))

        elif vec == "resnet18":
            from .features.resnet18 import build_image_features
            Xt, Xv, _ = build_image_features(df, (train_idx, test_idx))

        elif vec == "dino":
            from .features.dino import build_image_features

            Xt, Xv, _ = build_image_features(df,(train_idx, test_idx))

        else:
            raise ValueError("Unknown vectorizer")

        features_train.append(Xt)
        features_test.append(Xv)

    # ========================
    # FUSION
    # ========================

    # if early-fusion then X+X2 and one model, if late-fusion then two models

    if config["type"] == "early-fusion":
        # print(f"features_train {np.size(features_train[0])}")
        # print(f"features_train {np.size(features_train[1])}")
        # print(f"features_test {np.size(features_test[0])}")
        # print(f"features_test {np.size(features_test[1])}")
        X_train_final = np.hstack(features_train)
        X_test_final = np.hstack(features_test)

        features_train = [X_train_final]
        features_test = [X_test_final]

    # print(f"features_train {np.size(features_train[0])}")
    # print(f"features_test {np.size(features_test[0])}")

    # ========================
    # MODELS AND PREDICTIONS
    # ========================
    preds = []
    probas = []
    evaluations = {}
    y_preds = []

    for i, model_name in enumerate(config["models"]):
        best_threshold = 0.0
        thresholds = config.get("thresholds", [])
        Xtr = features_train[i]
        Xte = features_test[i]

        if model_name == "logistic":
            from .models.logistic import train_logistic
            model = train_logistic(Xtr, y_train, config["balanced_list"][i])

            #print(f"Best threshold={best_threshold:.2f} (train micro-F1={train_f1:.4f})")
            y_proba = get_positive_proba(model, Xte)
            if len(thresholds) > 0:
                for threshold in thresholds:
                    y_pred = (
                            y_proba > (threshold)
                    ).astype(int)
                    y_preds.append(y_pred)
                    print("predict sum:", y_pred.sum())
            else:
                train_proba = get_positive_proba(model, Xtr)

                best_threshold, train_f1 = find_best_threshold(
                    y_train,
                    train_proba
                )
                y_pred = (
                        y_proba > (best_threshold)
                ).astype(int)
                print(f"Best threshold={best_threshold:.2f} (train micro-F1={train_f1:.4f})")
                y_preds.append(y_pred)

                print("shape:", y_proba.shape)
                print("min:", y_proba.min())
                print("max:", y_proba.max())
                print("mean:", y_proba.mean())
                print("pred labels:", (y_proba > best_threshold).sum())
                print("true labels:", y_test.sum())
                print("predict sum:", y_pred.sum())
                print("true sum:", y_test.sum())

            '''
            #threshold = config["thresholds"][i] or 0.5
            #y_pred = (y_proba > threshold).astype(int)


            y_proba = get_positive_proba(model, Xte)
            print("shape:", y_proba.shape)
            print("min:", y_proba.min())
            print("max:", y_proba.max())
            print("mean:", y_proba.mean())
            print("pred labels:", (y_proba > threshold).sum())
            print("true labels:", y_test.sum())
            y_pred = (y_proba > threshold).astype(int)
            print("predict sum:", y_pred.sum())
            print("true sum:", y_test.sum())
            '''

        elif model_name == "svm":
            from .models.svm import train_svm
            model = train_svm(Xtr, y_train, config["balanced_list"][i], **clean_model_config(config, ["balanced"]))
            y_pred = model.predict(Xte)
            y_proba = model.predict(Xte)

        elif model_name == "random_forest":
            from .models.randomforest import train_random_forest
            print(f"Balanced: {config["balanced_list"][i]}")
            model = train_random_forest(Xtr, y_train, balanced=config["balanced_list"][i],
                                        **clean_model_config(config, ["balanced"]))
            #y_pred = model.predict(Xte)
            #y_proba = model.predict_proba(Xte)

            # print(f"Best threshold={best_threshold:.2f} (train micro-F1={train_f1:.4f})")
            y_proba = get_positive_proba(model, Xte)
            if len(thresholds) > 0:
                for threshold in thresholds:
                    y_pred = (
                            y_proba > (threshold)
                    ).astype(int)
                    y_preds.append(y_pred)
                    print("predict sum:", y_pred.sum())
            else:
                train_proba = get_positive_proba(model, Xtr)

                best_threshold, train_f1 = find_best_threshold(
                    y_train,
                    train_proba
                )
                print(f"Best threshold={best_threshold:.2f} (train micro-F1={train_f1:.4f})")
                y_pred = (
                        y_proba > (best_threshold)
                ).astype(int)
                y_preds.append(y_pred)

                print("shape:", y_proba.shape)
                print("min:", y_proba.min())
                print("max:", y_proba.max())
                print("mean:", y_proba.mean())
                print("pred labels:", (y_proba > best_threshold).sum())
                print("true labels:", y_test.sum())
                print("predict sum:", y_pred.sum())
                print("true sum:", y_test.sum())

        elif model_name == "mlp":
            from .models.mlp import train_mlp
            model = train_mlp(Xtr, y_train, **config)
            #y_pred = model.predict(Xte)
            #y_proba = model.predict_proba(Xte)

            # print(f"Best threshold={best_threshold:.2f} (train micro-F1={train_f1:.4f})")
            y_proba = get_positive_proba(model, Xte)
            if len(thresholds) > 0:
                for threshold in thresholds:
                    y_pred = (
                            y_proba > (threshold)
                    ).astype(int)
                    y_preds.append(y_pred)
                    print("predict sum:", y_pred.sum())
            else:
                train_proba = get_positive_proba(model, Xtr)

                best_threshold, train_f1 = find_best_threshold(
                    y_train,
                    train_proba
                )
                print(f"Best threshold={best_threshold:.2f} (train macro-F1={train_f1:.4f})")
                y_pred = (
                        y_proba > (best_threshold)
                ).astype(int)
                y_preds.append(y_pred)

                print("shape:", y_proba.shape)
                print("min:", y_proba.min())
                print("max:", y_proba.max())
                print("mean:", y_proba.mean())
                print("pred labels:", (y_proba > best_threshold).sum())
                print("true labels:", y_test.sum())
                print("predict sum:", y_pred.sum())
                print("true sum:", y_test.sum())


        else:
            raise ValueError("Unknown model")

        if len(config["models"]) == 1:
            for i, yps in enumerate(y_preds):
                dictio = evaluate(y_test, yps)
                dictio["predict sum"] = yps.sum()
                dictio["true sum"] = y_test.sum()
                dictio["best_threshold"] = best_threshold if best_threshold > 0.0 else thresholds[i] if len(thresholds) > i else 0.0
                #print("thresholds: ", i, thresholds[i])
                evaluations[f"none_{thresholds[i] if len(thresholds) > i else ""}"] = dictio
        preds.append(y_pred)
        probas.append(y_proba)

    # ========================
    # LATE FUSION (CLEAN VERSION)
    # ========================
    if config["type"] == "late-fusion":
        proba1, proba2 = probas
        proba1 = np.array(proba1)
        proba2 = np.array(proba2)

        # ========================
        # BASELINES
        # ========================
        y_or = np.logical_or.reduce(preds).astype(int)
        y_and = np.logical_and.reduce(preds).astype(int)
        y_avg = (np.mean(probas, axis=0) > 0.5).astype(int)

        evaluations["late-fusion-or"] = evaluate(y_test, y_or)
        evaluations["late-fusion-and"] = evaluate(y_test, y_and)
        evaluations["late-fusion-avg"] = evaluate(y_test, y_avg)

    return evaluations

def get_positive_proba(model, X):
    proba = model.predict_proba(X)

    if isinstance(proba, list):
        return np.column_stack([p[:, 1] for p in proba])

    proba = np.asarray(proba)

    if proba.ndim == 3:
        return proba[:, :, 1]

    return proba

from sklearn.metrics import f1_score, hamming_loss
import numpy as np

def find_best_threshold(y_true, y_proba,
                        thresholds=np.arange(0.05, 0.95, 0.05)):

    best_t = 0.5
    best_f1 = -1

    for t in thresholds:

        y_pred = (y_proba > t).astype(int)

        score = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ) - 0*hamming_loss(
            y_true,
            y_pred
        )

        if score > best_f1:
            best_f1 = score
            best_t = t

    return best_t, best_f1