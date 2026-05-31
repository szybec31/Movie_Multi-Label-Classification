# models/randomforest.py
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

def get_random_forest(balanced=True, **config):
    base_model = RandomForestClassifier(
        n_estimators = config["n_estimators"] if "n_estimators" in config else 200,
        max_depth = config["max_depth"] if "max_depth" in config else 5,
        max_features = config["max_features_rf"] if "max_features_rf" in config else 'sqrt',
        n_jobs=-1,
        random_state=42,
        class_weight='balanced' if balanced else None
    )

    model = OneVsRestClassifier(base_model)
    return model

def train_random_forest(X_train, y_train, balanced=True, **config):
    model = get_random_forest(balanced, **config)    

    model.fit(X_train, y_train)
    return model