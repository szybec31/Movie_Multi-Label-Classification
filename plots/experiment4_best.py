import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_best_systems(
    df,
    selection_metric="f1_samples_mean"
):
    """
    Select best model from each family using selection_metric
    and compare all metrics.
    """

    metrics = [
        ("f1_micro", "F1-micro"),
        ("f1_samples", "F1-samples"),
        ("recall_micro", "Recall-micro"),
        ("hamming", "Hamming loss"),
    ]

    groups = [
        ("text", None, "Text"),
        ("graphics", None, "Graphics"),
        ("early-fusion", None, "Early Fusion"),
        ("late-fusion", "late-fusion-or", "LF-OR"),
        ("late-fusion", "late-fusion-and", "LF-AND"),
        ("late-fusion", "late-fusion-avg", "LF-AVG"),
    ]

    selected_rows = []

    for type_name, experiment_name, label in groups:

        subset = df[df["type"] == type_name].copy()

        if experiment_name is not None:
            subset = subset[
                subset["experiment"] == experiment_name
            ]

        if subset.empty:
            continue

        best_idx = subset[selection_metric].idxmax()

        row = subset.loc[best_idx].copy()
        row["plot_label"] = label

        selected_rows.append(row)

    best_df = pd.DataFrame(selected_rows)

    # ----------------------------------
    # plotting
    # ----------------------------------

    x = np.arange(len(metrics))

    width = 0.13

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i, (_, row) in enumerate(best_df.iterrows()):

        means = []
        stds = []

        for metric, _ in metrics:
            means.append(row[f"{metric}_mean"])
            stds.append(row[f"{metric}_std"])

        ax.bar(
            x + (i - (len(best_df)-1)/2) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=row["plot_label"]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [label for _, label in metrics]
    )

    ax.set_ylabel("Score")
    ax.set_title(
        "Best Model from Each Approach Family\n"
        "(selected by F1-samples)"
    )

    ax.legend(
        title="Approach",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

    return best_df

if __name__ == "__main__":
    df = pd.read_csv("results_mean.csv")

    best_models = plot_best_systems(
        df,
        selection_metric="f1_samples_mean"
    )

    print(
        best_models[
            [
                "plot_label",
                "model1",
                "model2",
                "vectorizer1",
                "vectorizer2",
                "f1_micro_mean",
                "f1_samples_mean",
            ]
        ]
    )