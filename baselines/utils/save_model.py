import pandas as pd
import os

def save_model_info(config, results, elapsed_time, add: str = "", file_path: str = "results.csv"):

    rows = []

    for subtype, folds in results.items():

        for fold_result in folds:

            row = {
                "type": config["type"],
                "subtype": subtype if subtype != "none" else None,
                "add": add,
                "vectorizer1": str(config["vectorizers"][0]),
                "model1": str(config["models"][0]),
                "vectorizer2": str(config["vectorizers"][1]) if len(config["vectorizers"]) == 2 else None,
                "model2": str(config["models"][1]) if len(config["models"]) == 2 else None,
                "fold": fold_result["fold"],
                "time": elapsed_time,
                "best_threshold": fold_result.get("best_threshold", None),
            }

            row.update(fold_result)
            row["config"] = str(config)

            rows.append(row)

    df_new = pd.DataFrame(rows)

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(file_path, index=False)