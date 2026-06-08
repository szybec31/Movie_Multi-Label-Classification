import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_models_for_type(
    df,
    exp_type="text",
    metrics=None
):
    """
    One plot for:
        text
        graphics
        early-fusion

    X = metrics
    hue = model
    Y = metric value
    error bar = std
    """

    if metrics is None:
        metrics = [
            "f1_micro",
            "f1_samples",
            "recall_micro",
            "hamming"
        ]

    data = df[
        (df["type"] == exp_type)
        & (df["experiment"] == "default")
    ].copy()

    model_map = {
        "logistic": "LR",
        "random_forest": "RF",
        "mlp": "MLP"
    }

    models = [
        m for m in ["logistic", "random_forest", "mlp"]
        if m in data["model1"].values
    ]

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, model in enumerate(models):

        subset = data[data["model1"] == model]

        means = [
            subset[f"{metric}_mean"].iloc[0]
            for metric in metrics
        ]

        stds = [
            subset[f"{metric}_std"].iloc[0]
            for metric in metrics
        ]

        positions = x + (i - 1) * width

        plt.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=model_map.get(model, model)
        )

    plt.xticks(
        x,
        [m.replace("_", "\n").title() for m in metrics]
    )

    plt.ylabel("Score")
    plt.xlabel("Metric")

    plt.title(
        f"{exp_type.title()} Models Comparison"
    )

    plt.legend(title="Model")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv("results_mean.csv")

    plot_models_for_type(df, "text")

    plot_models_for_type(df, "graphics")

    plot_models_for_type(df, "early-fusion")