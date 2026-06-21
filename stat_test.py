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

CSV_PATH = "results_folds.csv"
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
    df["pca"] = df["pca"].fillna("none")

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

    if True or normality["normal"]:

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
        raise ValueError(f"Nie znaleziono Modelu {model_a}")

    if df_b.empty:
        raise ValueError(f"Nie znaleziono Modelu {model_b}")

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


def build_comparison_table(
    csv_path,
    models,
    metric="f1_samples",
    alpha=0.05,
    output_csv="pairwise_comparison.csv"
):

    df = load_results(csv_path)

    names = list(models.keys())

    wins = {n: 0 for n in names}
    losses = {n: 0 for n in names}
    equals = {n: 0 for n in names}

    matrix = pd.DataFrame(
        "-",
        index=names,
        columns=names
    )

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            name_a = names[i]
            name_b = names[j]

            result = compare_models(
                csv_path=csv_path,
                model_a=models[name_a],
                model_b=models[name_b],
                metric=metric
            )

            p_value = result["p_value"]

            scores_a, scores_b = prepare_paired_samples(
                select_model(df, models[name_a]),
                select_model(df, models[name_b]),
                metric
            )

            mean_a = scores_a.mean()
            mean_b = scores_b.mean()

            # ===================================
            # Significant difference
            # ===================================

            if p_value < alpha:

                if mean_a > mean_b:

                    matrix.loc[name_a, name_b] = f"↑ ({p_value:.3f})"
                    matrix.loc[name_b, name_a] = f"↓ ({p_value:.3f})"

                    wins[name_a] += 1
                    losses[name_b] += 1

                else:

                    matrix.loc[name_a, name_b] = f"↓ ({p_value:.3f})"
                    matrix.loc[name_b, name_a] = f"↑ ({p_value:.3f})"

                    wins[name_b] += 1
                    losses[name_a] += 1

            # ===================================
            # No significant difference
            # ===================================

            else:

                matrix.loc[name_a, name_b] = f"= ({p_value:.3f})"
                matrix.loc[name_b, name_a] = f"= ({p_value:.3f})"

                equals[name_a] += 1
                equals[name_b] += 1

    # ==========================================
    # Add summary columns
    # ==========================================

    matrix["W"] = [wins[n] for n in names]
    matrix["L"] = [losses[n] for n in names]
    matrix["E"] = [equals[n] for n in names]
    matrix["Score"] = matrix["W"] - matrix["L"]

    # matrix = matrix.sort_values(
    #     by=["Score", "W"],
    #     ascending=False
    # )

    matrix.to_csv(output_csv)

    print(f"\nSaved comparison table to: {output_csv}")

    return matrix

# ==========================================
# MODEL NAMES FOR TABLE
# ==========================================

MODELS = {
    "Text": {
        "type": "text",
        "experiment": "default",
        "vectorizer1": "distilbert",
        "model1": "mlp",
        "vectorizer2": "none",
        "model2": "none",
        "pca": "none"
    },
    "Graphics": {
        "type": "graphics",
        "experiment": "default",
        "vectorizer1": "resnet50",
        "model1": "mlp",
        "vectorizer2": "none",
        "model2": "none",
        "pca": "none",
    },
    "Early": {
        "type": "early-fusion",
        "experiment": "default",
        "vectorizer1": "distilbert",
        "model1": "mlp",
        "vectorizer2": "resnet50",
        "model2": "none",
        "pca": "none",
    },
    "Late OR": {
        "type": "late-fusion",
        "experiment": "late-fusion-or",
        "vectorizer1": "distilbert",
        "model1": "mlp",
        "vectorizer2": "resnet50",
        "model2": "mlp",
        "pca": 0.95,
    },
    "Late AND": {
        "type": "late-fusion",
        "experiment": "late-fusion-and",
        "vectorizer1": "distilbert",
        "model1": "mlp",
        "vectorizer2": "resnet50",
        "model2": "logistic",
        "pca": "none",
    },
    "Late AVG": {
        "type": "late-fusion",
        "experiment": "late-fusion-avg",
        "vectorizer1": "distilbert",
        "model1": "mlp",
        "vectorizer2": "resnet50",
        "model2": "random_forest",
        "pca": "none",
    },
}

if __name__ == "__main__":

    table = build_comparison_table(
        csv_path=CSV_PATH,
        models=MODELS,
        metric="f1_samples",
        output_csv="f1_samples_pairwise.csv"
    )

    print(table)