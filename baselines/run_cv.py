from .run_experiment import run_experiment
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def run_cv(df, y, n_splits=10, **config):

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

        for model_name, metrics in fold_results.items():

            if model_name not in all_results:
                all_results[model_name] = []

            row = {
                "fold": fold,
                **metrics
            }

            all_results[model_name].append(row)

    for model_name in all_results:
        print(f" ------- Info {model_name}: ------- ")
        for metric_name in all_results[model_name][0]:
            if metric_name == "fold":
                continue
            metric = []
            for fold in all_results[model_name]:
                metric.append(fold[metric_name])
            print(f"{metric_name}: {np.mean(metric):.3f}")

    return all_results