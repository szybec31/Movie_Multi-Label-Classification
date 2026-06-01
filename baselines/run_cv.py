from .run_experiment import run_experiment
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt
from .utils.save_model import save_fold_results, save_all_results

def run_cv(df, y, **config):

    n_splits = config.get("outer_cv", 10)

    X = df["title"]

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_results = {}

    for fold, (train_idx, test_idx) in enumerate(mskf.split(X, y)):

        print(f"Fold {fold+1}/{n_splits}")

        fold_results = run_experiment(
            df,
            y,
            split=(train_idx, test_idx),
            **config
        )

        
        ## save raw (row)
        save_fold_results(
            config=config,
            fold=fold,
            results=fold_results,
            file_path="results_folds.csv"
        )

        for model_name, metrics in fold_results.items():

            if model_name not in all_results:
                all_results[model_name] = []

            row = {
                "fold": fold,
                **metrics
            }

            all_results[model_name].append(row)

    
    for model_name in all_results:
        mean_results = {}
        print(f" ------- Info {model_name}: ------- ")
        for metric_name in all_results[model_name][0]:
            if metric_name == "fold":
                continue
            metric = []
            for fold in all_results[model_name]:
                metric.append(fold[metric_name])
            mean_results[f"{metric_name}_mean"] = np.mean(metric)
            mean_results[f"{metric_name}_std"] = np.std(metric)

            save_all_results(
                config=config,
                exp_name=model_name,
                mean_results=mean_results,
                file_path="results_mean.csv"
            )
    
    if config.get("all_threshold", False):
        thresholds = []
        f1_scores = []

        for threshold_name, folds in all_results.items():

            threshold = float(
                threshold_name.replace("threshold_", "")
            )

            mean_f1 = np.mean(
                [f["f1_macro"] for f in folds]
            )

            thresholds.append(threshold)
            f1_scores.append(mean_f1)

        plt.plot(thresholds, f1_scores)
        plt.xlabel("Threshold")
        plt.ylabel("Macro F1")
        plt.title("Threshold search")

    return all_results