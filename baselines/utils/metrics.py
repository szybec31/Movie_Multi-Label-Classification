# utils/metrics.py

from sklearn.metrics import f1_score, recall_score, hamming_loss

def evaluate(y_test, y_pred):
    return {
        "f1_micro": f1_score(y_test, y_pred, average='micro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_samples": f1_score(y_test, y_pred, average='samples'),
        "recall_micro": recall_score(y_test, y_pred, average='micro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "hamming": hamming_loss(y_test, y_pred),
        "avg_labels_true": y_test.sum(axis=1).mean(),
        "avg_labels_pred": y_pred.sum(axis=1).mean(),
        "predict sum": y_pred.sum(),
        "true sum": y_test.sum()
    }