import numpy as np
from sklearn.metrics import f1_score
from .utils.remake_config import clean_model_config
from .utils.metrics import evaluate
from .utils.clear import clean_text
from sklearn.metrics import f1_score, hamming_loss
import numpy as np

def run_experiment(df, y, split=None, **config):
    # Could be outdated... ========================
    # type: str = "text" or "graphics" or "early-fusion" or "late-fusion"
    # subtype: str = "text" or "title" or "overview" (only for type = "text")
    # vectorizers: list[str] = up to 2 vectorizers from list above
    # models: list[str] = up to 2 models from list above
    # balanced: bool = True or False
    # balanced_list: list[bool] = list of using balanced params in models (only for "late-fusion")
    # threshold: float = 0.2, 0.3, 0.5
    # thresholds: list[float] = 
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

    if "thresholds_text" not in config:
        config["thresholds_text"] = [None]

    if "thresholds_graphics" not in config:
        config["thresholds_graphics"] = [None]

    if "thresholds" not in config:
        if config["type"] in ["text", "early-fusion", "late-fusion"]:
            config["thresholds"] = [
                config["thresholds_text"],
                config["thresholds_graphics"]
            ]
        else:
            config["thresholds"] = [
                config["thresholds_graphics"],
                [None]
            ]
    else:
        config["thresholds"] = [
            config["thresholds"],
            config["thresholds"]
        ]

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
        X_train_final = np.hstack(features_train)
        X_test_final = np.hstack(features_test)

        features_train = [X_train_final]
        features_test = [X_test_final]

    # ========================
    # MODELS AND PREDICTIONS
    # ========================
    preds = []
    probas = []
    evaluations = {}
    y_preds = []

    for i, model_name in enumerate(config["models"]):
        best_threshold = 0.0
        Xtr = features_train[i]
        Xte = features_test[i]
        thresholds = config["thresholds"][i]

        # ========================
        #       BASE MODELS
        # ========================

        if model_name == "logistic":
            from .models.logistic import get_logistic
            base = get_logistic(config["balanced_list"][i])

        elif model_name == "svm":
            from .models.svm import get_svm
            base = get_svm(config["balanced_list"][i], **clean_model_config(config, ["balanced"]))

        elif model_name == "random_forest":
            from .models.randomforest import get_random_forest
            base = get_random_forest(balanced=config["balanced_list"][i],
                **clean_model_config(config, ["balanced"])
            )
            
        elif model_name == "mlp":
            from .models.mlp import get_mlp
            base = get_mlp(**config)

        else:
            raise ValueError("Unknown model")
        
        # ========================
        #      GRID SEARCH
        # ========================

        if config.get("use_threshold_grid", False) and "grid" in config and len(config.get("grid", [])) == 2:
            grid_params = config["grid"][i]
            if grid_params:

                from sklearn.model_selection import GridSearchCV
                from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
                from sklearn.metrics import make_scorer, f1_score

                scorer = make_scorer(
                    f1_score,
                    average="macro",
                    zero_division=0
                )

                inner_cv = MultilabelStratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=42
                )
                
                grid_search = GridSearchCV(
                    estimator = base,
                    param_grid = grid_params,
                    cv = inner_cv,
                    scoring = scorer,
                    n_jobs = -1
                )

                grid_search.fit(Xtr, y_train)
                #print(f"\n    Best params: {grid_search.best_params_}")
                #print(f"    Best CV MAE: {-grid_search.best_score_:.4f}")
                best_model = grid_search.best_estimator_

                best_model.fit(
                    Xtr,
                    y_train
                )

                y_proba = best_model.predict_proba(Xte)

                y_pred = (
                    y_proba > best_threshold
                ).astype(int)
            
            else:
                best_model = base
                best_model.fit(Xtr, y_train)

        else:
            best_model = base
            best_model.fit(Xtr, y_train)


        # ========================
        #       THRESHOLD
        # ========================

        if model_name in ["logistic", "random_forest", "mlp"]:
            #print(f"Best threshold={best_threshold:.2f} (train micro-F1={train_f1:.4f})")
            y_proba = get_positive_proba(best_model, Xte)
            if config.get("all_thresholds", False) and len(thresholds) > 0 and thresholds[0] is not None:
                
                for threshold in thresholds:
                    y_pred = (
                        y_proba > (threshold)
                    ).astype(int)
                    y_preds.append(y_pred)
                    print("predict sum:", y_pred.sum())
            else:
                train_proba = get_positive_proba(best_model, Xtr)

                best_threshold, train_f1 = find_best_threshold(
                    y_train,
                    train_proba,
                    thresholds
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

        elif model_name in ["svm"]:
            y_pred = best_model.predict(Xte)
            y_proba = best_model.predict(Xte)

        # ========================
        #       METRICS
        # ========================

        if len(config["models"]) == 1:
            for j, yps in enumerate(y_preds):
                dictio = evaluate(y_test, yps)
                dictio["best_threshold"] = best_threshold if best_threshold > 0.0 else thresholds[j] if len(thresholds) > j else 0.0
                #print("thresholds: ", i, thresholds[i])
                evaluations[f"threshold_{thresholds[j] if len(thresholds) > j else best_threshold}"] = dictio
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

def find_best_threshold(y_true, y_proba, thresholds: list[float] | None = None):

    if thresholds is None:
        thresholds=np.arange(0.05, 0.95, 0.05)

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