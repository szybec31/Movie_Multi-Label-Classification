from .run_experiment import run_experiment
from sklearn.model_selection import StratifiedKFold
import numpy as np

def run_cv(df, y, n_splits=10, **config):

    X = df["title"]
    y_strat = y.sum(axis=1)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_results = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_strat)):

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

    return all_results