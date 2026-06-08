from .run_experiment import run_experiment
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt
from .utils.save_model import save_fold_results, save_all_results
from .utils.get_subset import get_subset
import time
from collections import Counter
import json

def run_cv(df_, y_, **config):

    n_splits = config.get("outer_cv", 10)

    if config.get("use_subset", False):
        df, y = get_subset(df_, y_, config.get("subset_size", 0.2))
    else:
        df = df_
        y = y_

    X = df["title"]

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_results = {}

    for fold, (train_idx, test_idx) in enumerate(mskf.split(X, y)):

        print(f"Fold {fold+1}/{n_splits}")

        fold_start = time.time()

        fold_results = run_experiment(
            df,
            y,
            split=(train_idx, test_idx),
            **config
        )

        fold_time = time.time() - fold_start

        
        ## save raw (row)
        save_fold_results(
            config=config,
            fold=fold,
            results=fold_results,
            fold_time=fold_time,
            file_path="results_folds.csv"
        )

        for model_name, metrics in fold_results.items():

            if model_name not in all_results:
                all_results[model_name] = []

            row = {
                "fold": fold,
                "fold_time": fold_time,
                **metrics
            }

            all_results[model_name].append(row)

    
    for model_name in all_results:
        mean_results = {}
        print(f" ------- Info {model_name}: ------- ")
        for metric_name in all_results[model_name][0]:
            
            if metric_name == "fold":
                continue

            if metric_name == "best_params":
                params_list = [
                    json.dumps(f["best_params"], sort_keys=True)
                    for f in all_results[model_name]
                    if f["best_params"]
                ]
                mean_results["all_best_params"] = str(
                    Counter(params_list)
                )
                continue
            
            metric = []
            for fold in all_results[model_name]:
                metric.append(fold[metric_name])

            if metric_name == "fold_time":
                mean_results["fold_time_sum"] = np.sum(metric)
            else:
                mean_results[f"{metric_name}_mean"] = np.mean(metric)
                mean_results[f"{metric_name}_std"] = np.std(metric)

        if config.get("save_more_metadata", False):
            mean_results["n_samples"] = len(df)
            mean_results["n_labels"] = y.shape[1]
            mean_results["outer_cv"] = config["outer_cv"]
            mean_results["inner_cv"] = config["inner_cv"]
            mean_results["use_subset"] = config.get("use_subset", False)
            mean_results["subset_size"] = config.get("subset_size", 1.0)

        save_all_results(
            config=config,
            exp_name=model_name,
            mean_results=mean_results,
            file_path="results_mean.csv"
        )
    
    if config.get("all_thresholds", False):
        print(" -- !!! -- ")
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
        if config.get("plot_show", False):
            plt.show()

    return all_results