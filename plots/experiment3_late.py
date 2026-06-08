import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")


MODEL_NAMES = {
    "logistic": "LR",
    "random_forest": "RF",
    "mlp": "MLP"
}


def plot_late_fusion_heatmap(
    df,
    metric="f1_micro"
):
    """
    metric:
        f1_micro
        recall_micro
        hamming
    """

    data = df[
        df["type"] == "late-fusion"
    ].copy()

    fusion_order = [
        "late-fusion-or",
        "late-fusion-and",
        "late-fusion-avg"
    ]

    fusion_names = {
        "late-fusion-or": "OR",
        "late-fusion-and": "AND",
        "late-fusion-avg": "AVG"
    }

    data["fusion"] = data["experiment"]

    data["combo"] = (
        data["model1"].map(MODEL_NAMES)
        + "+"
        + data["model2"].map(MODEL_NAMES)
    )

    pivot = data.pivot(
        index="combo",
        columns="fusion",
        values=f"{metric}_mean"
    )

    row_order = [
        "LR+LR",
        "LR+RF",
        "LR+MLP",
        "RF+LR",
        "RF+RF",
        "RF+MLP",
        "MLP+LR",
        "MLP+RF",
        "MLP+MLP"
    ]

    pivot = pivot.reindex(row_order)
    pivot = pivot[fusion_order]

    pivot.columns = [
        fusion_names[c]
        for c in pivot.columns
    ]

    plt.figure(figsize=(6, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar=True
    )

    plt.title(
        metric.replace("_", " ").title()
    )

    plt.xlabel("Fusion Type")
    plt.ylabel("Model Combination")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv("results_mean.csv")

    plot_late_fusion_heatmap(
        df,
        metric="f1_samples"
    )

    plot_late_fusion_heatmap(
        df,
        metric="recall_micro"
    )

    plot_late_fusion_heatmap(
        df,
        metric="hamming"
    )