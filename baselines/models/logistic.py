# models/logistic.py

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

def train_logistic(X_train, y_train, balanced=True):
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced' if balanced else None
        )
    )
    model.fit(X_train, y_train)
    return model