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
        "mean_a": np.mean(scores_a),
        "mean_b": np.mean(scores_b),
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

def print_significance_matrix(summary, model_names, metric = "f1_macro"):

    matrix = pd.DataFrame(
        "-",
        index=model_names,
        columns=model_names
    )

    for model in model_names:
        matrix.loc[model, model] = "X"

    for row in summary:

        a = row["model_a"]
        b = row["model_b"]

        if row["p_value"] < ALPHA:

            if row["mean_a"] > row["mean_b"]:
                matrix.loc[a, b] = "W"
                matrix.loc[b, a] = "L"
            else:
                matrix.loc[a, b] = "L"
                matrix.loc[b, a] = "W"

        else:
            matrix.loc[a, b] = "="
            matrix.loc[b, a] = "="

    print("\n" + "=" * 70)
    print(f"SIGNIFICANCE MATRIX -- FOR {metric}")
    print("=" * 70)

    print("\nLegend:")
    print("W = significantly better")
    print("L = significantly worse")
    print("= = no significant difference")
    print("X = same model\n")

    print(matrix)

    return matrix

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

def run(metric = "f1_macro"):
    summary = []

    model_names = [
        "text",
        "graphics",
        "early-fusion",
        "late-fusion-avg",
        "late-fusion-or",
        "late-fusion-and"
    ]

    models = [
        {
            "type": "text",
            "vectorizer1": "distilbert",
            "model1": "random_forest",
            "vectorizer2": "none",
            "model2": "none"
        },
        {
            "type": "graphics",
            "vectorizer1": "resnet50",
            "model1": "random_forest",
            "vectorizer2": "none",
            "model2": "none"
        },
        {
            "type": "early-fusion",
            #"subtype": "late-fusion-or",
            "vectorizer1": "distilbert",
            "model1": "random_forest",
            "vectorizer2": "resnet50",
            #"model2": "logistic"
        },
        {
            "type": "late-fusion",
            "subtype": "late-fusion-avg",
            "vectorizer1": "distilbert",
            "model1": "logistic",
            "vectorizer2": "resnet50",
            "model2": "random_forest"
        },
        {
            "type": "late-fusion",
            "subtype": "late-fusion-or",
            "vectorizer1": "distilbert",
            "model1": "mlp",
            "vectorizer2": "resnet50",
            "model2": "mlp"
        },
        {
            "type": "late-fusion",
            "subtype": "late-fusion-and",
            "vectorizer1": "distilbert",
            "model1": "logistic",
            "vectorizer2": "resnet50",
            "model2": "random_forest"
        }
    ]

    for i in range(5):
        for j in range(i+1, 6):
            results = compare_models(
                csv_path=CSV_PATH,
                model_a=models[i],
                model_b=models[j],
                metric=metric
            )

            summary.append({
                "model_a": model_names[i],
                "model_b": model_names[j],
                "p_value": results["p_value"],
                "effect_size": results["effect_size"],
                "mean_a": results["mean_a"],
                "mean_b": results["mean_b"]
            })

    wins = {name: 0 for name in model_names}
    losses = {name: 0 for name in model_names}
    ties = {name: 0 for name in model_names}

    for row in summary:

        a = row["model_a"]
        b = row["model_b"]

        if row["p_value"] < ALPHA:

            if row["mean_a"] > row["mean_b"]:
                wins[a] += 1
                losses[b] += 1
            else:
                wins[b] += 1
                losses[a] += 1

        else:
            ties[a] += 1
            ties[b] += 1

    print("\n" + "="*70)
    print("FINAL STATISTICAL SUMMARY")
    print("="*70)

    ranking = sorted(
        model_names,
        key=lambda m: wins[m],
        reverse=True
    )

    for model in ranking:
        print(
            f"{model:20s}"
            f"Wins={wins[model]:2d} "
            f"Losses={losses[model]:2d} "
            f"Ties={ties[model]:2d}"
        )

    return summary, model_names

if __name__ == "__main__":

    summaries = []
    metrics = ["f1_micro", "f1_macro", "b1", "recall_micro", "hamming"]

    for metric in metrics:

        metric = "f1_macro"

        summary, model_names = run(metric=metric)
        summaries.append(summary)

    for summary, metric in zip(summaries, metrics):

        matrix = print_significance_matrix(
            summary,
            model_names,
            metric = metric
        )