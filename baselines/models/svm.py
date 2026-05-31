from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

def train_svm(X_train, y_train, balanced=True, **config):
    model = OneVsRestClassifier(
        LinearSVC(
        )
    )
    model.fit(X_train, y_train)
    return model