def run_experiment(df, y, config, split=None):

    # input
    if config["input"] == "text":
        X = df["title"].fillna('') + " " + df["overview"].fillna('')
    else:
        X = df[config["input"]]

    # split (jeśli nie podany → default)
    if split is None:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        train_idx, test_idx = split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    # features
    from .features.tfidf import build_tfidf
    X_train, X_test, _ = build_tfidf(X_train, X_test)

    # model
    # model selection

    if config["model"] == "logistic":
        from .models.logistic import train_logistic
        model = train_logistic(X_train, y_train, config["balanced"])

    elif config["model"] == "svm":
        from .models.svm import train_svm
        model = train_svm(X_train, y_train, config["balanced"])

    # predict
    if config["model"] == "logistic":
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba > config["threshold"]).astype(int)

    elif config["model"] == "svm":
        y_pred = model.predict(X_test)

    # metrics
    from .utils.metrics import evaluate
    results = evaluate(y_test, y_pred)

    return results