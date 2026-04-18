from .tf_idf import model as tfidf
from sklearn.metrics import f1_score, classification_report, recall_score, hamming_loss
import os

def run_model(type, df, y, opt = ["text", 0.5, True]):
    df["text"] = df["title"].fillna('') + " " + df["overview"].fillna('')

    if type == "tfidf":
        return tfidf(df, y, opt[0], opt[1], opt[2])

def save_model_info(dir_name, file_name, y_test, y_pred, y_label):
    os.makedirs(dir_name, exist_ok=True)
    file_path = os.path.join(dir_name, file_name)

    output = []
    output.append(f"F1 micro: {f1_score(y_test, y_pred, average='micro')}")
    output.append(f"F1 macro: {f1_score(y_test, y_pred, average='macro')}")
    output.append(f"Recall weighted: {recall_score(y_test, y_pred, average='weighted')}")
    output.append(f"Avg labels per sample (true): {y_test.sum(axis=1).mean()}")
    output.append(f"Avg labels per sample (pred): {y_pred.sum(axis=1).mean()}")
    output.append(f"Hamming Loss: {hamming_loss(y_test, y_pred)}")
    output.append("\nClassification report:\n")
    output.append(classification_report(y_test, y_pred, target_names=y_label, zero_division=0))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))