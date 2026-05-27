from itertools import product
import time
from .run_cv import run_cv
from .utils.save_model import save_model_info
from .utils.get_subset import get_subset

TUNING_CV = 3 # Quick CV for tuning
TOP_K = 3 # Number of best param combination per model
FINAL_CV = 10 # Final CV (work on TOP_K number of param)
FRAC = 0.2 # 10% of dataset

def get_mean_metrics(results, subtype="none"):

    folds = results[subtype]

    metric_keys = [
        "f1_micro",
        "f1_macro",
        "b1",
        "recall_micro",
        "hamming",
        "avg_labels_true",
        "avg_labels_pred"
    ]

    metrics = {}

    for key in metric_keys:

        values = [
            f[key]
            for f in folds
            if key in f
        ]

        if len(values) == 0:
            metrics[key] = None
        else:
            metrics[key] = sum(values) / len(values)

    return metrics

def print_best_result(best_result, model_name):
    print("\n" + "=" * 80)
    print(f"BEST CONFIG FOR: {model_name}")
    print("=" * 80)

    print("\nCONFIG:")
    print(best_result["config"])

    print("\nPARAMS:")
    print(best_result["params"])

    print("\nMETRICS:")

    for k, v in best_result["metrics"].items():

        if v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {v:.4f}")

    print("=" * 80 + "\n")

def tuning_text(df_o, y_o):

    df, y = get_subset(df_o, y_o, FRAC)

    best_results = {
        "logistic": [],
        "svm": [],
        "random_forest": [],
        "mlp": []
    }
    vectorizers = ["distilbert"]
    models = [
        "logistic",
        "svm",
        "random_forest",
        "mlp"
    ]

    param_grids = {
        "logistic": {
            "balanced": [True, False],
            "threshold": [0.2, 0.3, 0.4, 0.5]
        },

        "svm": {
            "balanced": [True, False],
            "max_iter_svm": [3000, 5000]
        },

        "random_forest": {
            "balanced": [True, False],
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "max_features_rf": ["sqrt", 0.8]
        },

        "mlp": {
            "hidden_layer_sizes": [
                (256, 128),
                (512, 256)
            ],
            "learning_rate_init": [
                0.001,
                0.0005
            ],
            "batch_size": [32, 64],
            "max_iter": [40, 80]
        }
    }

    for vect in vectorizers:
        for model in models:
            grid = param_grids[model]
            keys = list(grid.keys())
            values = list(grid.values())

            for combo in product(*values):
                params = dict(zip(keys, combo))
                config = {
                    "type": "text",
                    "vectorizers": [vect],
                    "models": [model],
                    **params
                }

                print("=" * 80)
                print(config)

                start = time.time()
                results = run_cv(df, y, TUNING_CV, **config)
                end = time.time()

                add = (
                    f"Tuning | "
                    f"{params}"
                )

                save_model_info(config, results, end - start, add=add, file_path="tuning.csv")
                metrics = get_mean_metrics(results)
                score = metrics["f1_macro"]

                candidate = {
                    "score": score,
                    "config": config,
                    "params": params,
                    "metrics": metrics
                }

                best_results[model].append(candidate)

                best_results[model] = sorted(
                    best_results[model],
                    key=lambda x: x["score"],
                    reverse=True
                )[:TOP_K]

    print("\n\nFinal best results:\n")
    for model_name, candidates in best_results.items():

        print(f"\nFINAL TRAINING FOR: {model_name}")

        for i, result in enumerate(candidates):

            print(f"\nTOP {i+1}/{TOP_K}")

            print_best_result(result, model_name)

            config = result["config"]

            start = time.time()

            final_results = run_cv(
                df_o,
                y_o,
                FINAL_CV,
                **config
            )

            end = time.time()

            save_model_info(
                config,
                final_results,
                end - start,
                add=f"FINAL_TOP_{i+1} | {result["params"]}",
                file_path="tuning_final.csv"
            )


def tuning_graphics(df_o, y_o):

    df, y = get_subset(df_o, y_o, FRAC)

    best_results = {
        "logistic": [],
        "random_forest": [],
        "mlp": []
    }
    vectorizers = ["resnet50"]
    models = [
        "logistic",
        "random_forest",
        "mlp"
    ]

    param_grids = {
        "logistic": {
            "balanced": [True, False],
            "threshold": [0.2, 0.3, 0.4, 0.5]
        },

        "random_forest": {
            "balanced": [True, False],
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "max_features_rf": ["sqrt", 0.8]
        },

        "mlp": {
            "hidden_layer_sizes": [
                (256, 128),
                (512, 256)
            ],
            "learning_rate_init": [
                0.001,
                0.0005
            ],
            "batch_size": [32, 64],
            "max_iter": [40, 80]
        }
    }

    for vect in vectorizers:
        for model in models:
            grid = param_grids[model]
            keys = list(grid.keys())
            values = list(grid.values())

            for combo in product(*values):
                params = dict(zip(keys, combo))
                config = {
                    "type": "graphics",
                    "vectorizers": [vect],
                    "models": [model],
                    **params
                }

                print("=" * 80)
                print(config)

                start = time.time()
                results = run_cv(df, y, TUNING_CV, **config)
                end = time.time()

                add = (
                    f"Tuning | "
                    f"{params}"
                )

                save_model_info(config, results, end - start, add=add, file_path="tuning.csv")
                metrics = get_mean_metrics(results)
                score = metrics["f1_macro"]

                candidate = {
                    "score": score,
                    "config": config,
                    "params": params,
                    "metrics": metrics
                }

                best_results[model].append(candidate)

                best_results[model] = sorted(
                    best_results[model],
                    key=lambda x: x["score"],
                    reverse=True
                )[:TOP_K]

    print("\n\nFinal best results:\n")
    for model_name, candidates in best_results.items():

        print(f"\nFINAL TRAINING FOR: {model_name}")

        for i, result in enumerate(candidates):

            print(f"\nTOP {i+1}/{TOP_K}")

            print_best_result(result, model_name)

            config = result["config"]

            start = time.time()

            final_results = run_cv(
                df_o,
                y_o,
                FINAL_CV,
                **config
            )

            end = time.time()

            save_model_info(
                config,
                final_results,
                end - start,
                add=f"FINAL_TOP_{i+1} | {result["params"]}",
                file_path="tuning_final.csv"
            )


def tuning_early_fusion(df, y):
    best_results = {}
    text_vectorizers = ["distilbert"]
    image_vectorizers = ["resnet50"]

    models = [
        "logistic",
        "random_forest",
        "mlp"
    ]

    for text_vect in text_vectorizers:
        for image_vect in image_vectorizers:
            for model in models:

                config = {
                    "type": "early-fusion",
                    "vectorizers": [
                        text_vect,
                        image_vect
                    ],
                    "models": [model],
                    "balanced": True,
                    "max_features_tfidf": 20000,
                    "max_iter": 40,
                    "learning_rate_init": 0.001,
                    "max_depth": 20
                }

                print("=" * 80)
                print(config)

                start = time.time()
                results = run_cv(df, y, 10, **config)

                end = time.time()
                add = (
                    f"Tuning early-fusion | "
                    f"{text_vect}+{image_vect} | "
                    f"{model}"
                )

                save_model_info(config, results, end - start, add=add)



def tuning_late_fusion(df, y):
    best_results = {}
    text_vectorizers = ["distilbert"]
    image_vectorizers = ["resnet50"]

    text_models = [
        "logistic",
        "svm",
        "random_forest",
        "mlp"
    ]

    image_models = [
        "logistic",
        "random_forest",
        "mlp"
    ]

    for text_vect in text_vectorizers:
        for image_vect in image_vectorizers:
            for text_model in text_models:
                for image_model in image_models:

                    config = {
                        "type": "late-fusion",
                        "vectorizers": [
                            text_vect,
                            image_vect
                        ],
                        "models": [
                            text_model,
                            image_model
                        ],
                        "balanced_list": [
                            True,
                            True
                        ],
                        "max_features_tfidf": 20000,
                        "max_iter": 40,
                        "learning_rate_init": 0.001,
                        "max_depth": 20
                    }

                    print("=" * 80)
                    print(config)

                    start = time.time()
                    results = run_cv(df, y, 10, **config)

                    end = time.time()
                    add = (
                        f"Tuning late-fusion | "
                        f"{text_vect}+{image_vect} | "
                        f"{text_model}+{image_model}")

                    save_model_info(config, results, end - start, add=add)


def run_grid_search(df, y, test_type):
    if test_type == "text":
        tuning_text(df, y)

    elif test_type == "graphics":
        tuning_graphics(df, y)

    elif test_type == "early-fusion":
        tuning_early_fusion(df, y)

    elif test_type == "late-fusion":
        tuning_late_fusion(df, y)

    else:
        raise ValueError("Unknown tuning type")