def run_experiment(df, y, config, split=None):
    # ========================
    # TYPE ("text" or "graphics" or "early-fusion" or "late-fusion")
    # ========================
    if config["type"] == "graphics":
        X = df["poster_path"]
        if config["vectorizer"] not in ["resnet18", "resnet50"]:
            raise ValueError("Wrong vectorizer to chosen type")
            config["vectorizer"] = "resnet18"


    elif config["type"] == "text":
        if config["vectorizer"] not in ["tfidf", "bert"]:
            raise ValueError("Wrong vectorizer to chosen type")
            config["vectorizer"] = "tfidf"
        # ========================
        # SUBTYPE (for text only)
        # ========================
        if config["subtype"] == "text":
            X = df["title"].fillna('') + " " + df["overview"].fillna('')
        else:
            X = df[config["subtype"]]

    else:
        raise ValueError("Unknown type")
    


    # ========================
    # SPLIT
    # ========================
    if split is None:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        train_idx, test_idx = split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


    # ========================
    # FEATURES / VECTORIZER
    # ========================
    if config["vectorizer"] == "tfidf":
        from .features.tfidf import build_tfidf
        X_train, X_test, _ = build_tfidf(X_train, X_test)

    elif config["vectorizer"] == "resnet18":
        from .features.resnet18 import build_image_features
        X_train, X_test, _ = build_image_features(df, split)
    
    elif config["vectorizer"] == "resnet50":
        from .features.resnet50 import build_image_features
        X_train, X_test, _ = build_image_features(df, split)

    else:
        raise ValueError("Unknown vectorizer")


    # ========================
    # MODEL
    # ========================
    if config["model"] == "logistic":
        from .models.logistic import train_logistic
        model = train_logistic(X_train, y_train, config["balanced"])

    elif config["model"] == "svm":
        from .models.svm import train_svm
        model = train_svm(X_train, y_train, config["balanced"])
    
    else:
        raise ValueError("Unknown model")

    # ========================
    # PREDICT
    # ========================
    if config["model"] == "logistic":
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba > config["threshold"]).astype(int)

    elif config["model"] == "svm":
        y_pred = model.predict(X_test)

    # ========================
    # METRICS
    # ========================
    from .utils.metrics import evaluate
    results = evaluate(y_test, y_pred)

    return results