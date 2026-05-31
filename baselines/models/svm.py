from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

def get_svm(balanced=True, **config):
    model = OneVsRestClassifier(
        LinearSVC(
            class_weight='balanced' if balanced else None,
            max_iter=config.get("max_iter_svm", 1000)
        )
    )

    return model

def train_svm(X_train, y_train, balanced=True, **config):
    model = get_svm(balanced, **config)

    model.fit(X_train, y_train)
    return model