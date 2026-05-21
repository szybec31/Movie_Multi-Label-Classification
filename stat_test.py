import pandas as pd
import numpy as np

from scipy.stats import (
    shapiro,
    ttest_rel,
    wilcoxon
)

# =========================================================
# KONFIGURACJA
# =========================================================

CSV_PATH = "results.csv"
ALPHA = 0.05

# Dostępne metryki:
# f1_micro
# f1_macro
# b1
# recall_micro
# hamming
# avg_labels_true
# avg_labels_pred


# =========================================================
# WCZYTANIE DANYCH
# =========================================================

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ujednolicenie wartości None/NaN
    df["vectorizer2"] = df["vectorizer2"].fillna("none")
    df["model2"] = df["model2"].fillna("none")

    return df


# =========================================================
# WYBÓR MODELU
# =========================================================

def select_model(
    df: pd.DataFrame,
    model_config: dict
) -> pd.DataFrame:

    filtered = df.copy()

    for key, value in model_config.items():
        filtered = filtered[filtered[key] == value]

    return filtered.sort_values("fold")


# =========================================================
# PRZYGOTOWANIE PAR
# =========================================================

def prepare_paired_samples(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str
):

    merged = pd.merge(
        df_a[["fold", metric]],
        df_b[["fold", metric]],
        on="fold",
        suffixes=("_A", "_B")
    )

    if len(merged) != 10:
        raise ValueError(
            f"Oczekiwano 10 foldów, znaleziono {len(merged)}"
        )

    scores_a = merged[f"{metric}_A"].values
    scores_b = merged[f"{metric}_B"].values

    return scores_a, scores_b


# =========================================================
# TEST NORMALNOŚCI
# =========================================================

def check_normality(
    scores_a,
    scores_b
):

    differences = scores_a - scores_b

    stat, p_value = shapiro(differences)

    return {
        "statistic": stat,
        "p_value": p_value,
        "normal": p_value > ALPHA
    }


# =========================================================
# EFFECT SIZE
# =========================================================

def cohens_d_paired(scores_a, scores_b):

    diff = scores_a - scores_b

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    return mean_diff / std_diff


# =========================================================
# TEST STATYSTYCZNY
# =========================================================

def run_statistical_test(
    scores_a,
    scores_b
):

    normality = check_normality(scores_a, scores_b)

    if normality["normal"]:

        test_name = "Paired t-test"

        stat, p_value = ttest_rel(scores_a, scores_b)

        effect_size = cohens_d_paired(scores_a, scores_b)

    else:

        test_name = "Wilcoxon signed-rank test"

        stat, p_value = wilcoxon(scores_a, scores_b)

        effect_size = None

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "effect_size": effect_size,
        "normality": normality
    }


# =========================================================
# RAPORT
# =========================================================

def print_report(
    model_a,
    model_b,
    metric,
    scores_a,
    scores_b,
    results
):

    print("\n" + "=" * 60)
    print("PORÓWNANIE MODELI")
    print("=" * 60)

    print("\nMODEL A")
    for k, v in model_a.items():
        print(f"{k}: {v}")

    print("\nMODEL B")
    for k, v in model_b.items():
        print(f"{k}: {v}")

    print("\nMETRYKA:", metric)

    print("\nWYNIKI FOLDÓW")
    for i, (a, b) in enumerate(zip(scores_a, scores_b), start=1):
        print(f"Fold {i}: A={a:.4f} | B={b:.4f}")

    print("\n" + "-" * 60)
    print("TEST NORMALNOŚCI (Shapiro-Wilk)")
    print("-" * 60)

    print(
        f"Statistic = {results['normality']['statistic']:.6f}"
    )
    print(
        f"p-value   = {results['normality']['p_value']:.6f}"
    )

    if results["normality"]["normal"]:
        print("Wniosek: rozkład różnic jest normalny")
    else:
        print("Wniosek: brak normalności rozkładu różnic")

    print("\n" + "-" * 60)
    print("TEST STATYSTYCZNY")
    print("-" * 60)

    print(f"Test: {results['test']}")
    print(f"Statistic = {results['statistic']:.6f}")
    print(f"p-value   = {results['p_value']:.6f}")

    if results["effect_size"] is not None:
        print(f"Cohen's d = {results['effect_size']:.6f}")

    print("\n" + "-" * 60)

    if results["p_value"] < ALPHA:
        print(
            f"WYNIK ISTOTNY STATYSTYCZNIE "
            f"(p < {ALPHA})"
        )
    else:
        print(
            f"BRAK ISTOTNOŚCI STATYSTYCZNEJ "
            f"(p >= {ALPHA})"
        )

    print("=" * 60)


# =========================================================
# GŁÓWNA FUNKCJA
# =========================================================

def compare_models(
    csv_path,
    model_a,
    model_b,
    metric
):

    df = load_results(csv_path)

    df_a = select_model(df, model_a)
    df_b = select_model(df, model_b)

    if df_a.empty:
        raise ValueError("Nie znaleziono Modelu A")

    if df_b.empty:
        raise ValueError("Nie znaleziono Modelu B")

    scores_a, scores_b = prepare_paired_samples(
        df_a,
        df_b,
        metric
    )

    results = run_statistical_test(
        scores_a,
        scores_b
    )

    print_report(
        model_a,
        model_b,
        metric,
        scores_a,
        scores_b,
        results
    )

    return results


# =========================================================
# PRZYKŁAD UŻYCIA
# =========================================================

if __name__ == "__main__":

    model_A = {
        "type": "text",
        "vectorizer1": "distilbert",
        "model1": "mlp",
        "vectorizer2": "none",
        "model2": "none"
    }

    model_B = {
        "type": "graphics",
        "vectorizer1": "resnet50",
        "model1": "random_forest",
        "vectorizer2": "none",
        "model2": "none"
    }

    compare_models(
        csv_path=CSV_PATH,
        model_a=model_A,
        model_b=model_B,
        metric="f1_micro"
    )