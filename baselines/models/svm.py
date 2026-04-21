from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

def train_svm(X_train, y_train, balanced=True):
    model = OneVsRestClassifier(
        LinearSVC(class_weight='balanced' if balanced else None)
    )
    model.fit(X_train, y_train)
    return model