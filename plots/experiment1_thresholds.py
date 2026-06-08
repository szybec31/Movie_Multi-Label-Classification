import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_thresholds(
    df,
    modality="text",
    metric="f1_micro"
):
    """
    modality: text or graphics
    metric: f1_micro, f1_samples, recall_micro, hamming ...
    """

    data = df[
        (df["type"] == modality)
        & (df["experiment"].str.startswith("threshold_"))
    ].copy()

    data["threshold"] = (
        data["experiment"]
        .str.replace("threshold_", "", regex=False)
        .astype(float)
    )

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    plt.figure(figsize=(8, 5))

    for model in sorted(data["model1"].unique()):

        subset = (
            data[data["model1"] == model]
            .sort_values("threshold")
        )

        x = subset["threshold"]
        y = subset[mean_col]
        s = subset[std_col]

        # plt.plot(
        #     x,
        #     y,
        #     marker="o",
        #     linewidth=2,
        #     label=model
        # )

        plt.errorbar(
            x,
            y,
            yerr=s,
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=3,
            capsize=2,
            elinewidth=1,
            label=model
        )

    plt.xlabel("Threshold")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(
        f"{metric.replace('_', ' ').title()} vs Threshold ({modality})"
    )
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("results_mean.csv")
    plot_thresholds(
        df,
        modality="text",
        metric="f1_samples"
    )
    plot_thresholds(
        df,
        modality="graphics",
        metric="f1_samples"
    )