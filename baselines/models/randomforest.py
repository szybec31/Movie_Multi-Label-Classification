# models/randomforest.py
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, balanced=True):
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        class_weight='balanced' if balanced else None
    )

    model = OneVsRestClassifier(base_model)

    model.fit(X_train, y_train)
    return model