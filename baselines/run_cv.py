from .run_experiment import run_experiment
from sklearn.model_selection import StratifiedKFold
import numpy as np

def run_cv(df, y, config, n_splits=5):
    X = df["title"] # nie ma znaczenia kolumna, gdyz StratifiedKFold.split zwraca i tak tylko id's

    # pseudo-stratyfikacja (ważne!)
    y_strat = y.sum(axis=1)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_strat)):
        print(f"Fold {fold+1}/{n_splits}")

        results = run_experiment(
            df,
            y,
            config,
            split=(train_idx, test_idx)
        )

        all_results.append(results)

    # agregacja
    avg_results = {
        key: np.mean([r[key] for r in all_results])
        for key in all_results[0]
    }

    std_results = {
        key: np.std([r[key] for r in all_results])
        for key in all_results[0]
    }

    return avg_results, std_results