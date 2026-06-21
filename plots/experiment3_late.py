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
        (df["pca"] != 0.95) &
        (df["type"] == "late-fusion")
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

    pivot_mean = data.pivot(
    index="combo",
    columns="fusion",
    values=f"{metric}_mean"
)

    pivot_std = data.pivot(
        index="combo",
        columns="fusion",
        values=f"{metric}_std"
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

    pivot_mean = pivot_mean.reindex(row_order)
    pivot_mean = pivot_mean[fusion_order]

    pivot_std = pivot_std.reindex(row_order)
    pivot_std = pivot_std[fusion_order]

    annot = pivot_mean.copy().astype(str)

    for r in pivot_mean.index:
        for c in pivot_mean.columns:

            mean_val = pivot_mean.loc[r, c]
            std_val = pivot_std.loc[r, c]

            annot.loc[r, c] = (
                f"{mean_val:.3f}\n({std_val:.3f})"
            )
            
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        pivot_mean,
        annot=annot,
        fmt="",
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
        metric="avg_labels_pred"
    )