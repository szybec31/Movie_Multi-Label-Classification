from itertools import product
import time
from .run_cv import run_cv
from .utils.save_model import save_model_info


def tuning_text(df, y):
    vectorizers = ["tfidf", "distilbert"]
    models = [
        "logistic",
        "svm",
        "random_forest",
        "mlp"
    ]

    param_grids = {

        "logistic": {
            "balanced": [True, False]
        },

        "svm": {
            "balanced": [True, False],
            "max_iter_svm": [3000, 5000]
        },

        "random_forest": {
            "balanced": [True, False],
            "n_estimators": [100, 200],
            "max_depth": [10, 20]
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
                    "max_features_tfidf": 20000,
                    "ngram_range_tfidf": (1, 2),
                    **params
                }

                print("=" * 80)
                print(config)

                start = time.time()
                results = run_cv(df, y, 10, **config)
                end = time.time()

                add = (
                    f"Tuning text | "
                    f"{vect} | "
                    f"{model} | "
                    f"{params}"
                )

                save_model_info(config, results, end - start, add=add)


def tuning_graphics(df, y):
    vectorizers = ["resnet18", "resnet50"]
    models = [
        "logistic",
        "random_forest",
        "mlp"
    ]

    param_grids = {
        "logistic": {
            "balanced": [True, False]
        },

        "random_forest": {
            "balanced": [True, False],
            "n_estimators": [100, 200],
            "max_depth": [10, 20]
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
                results = run_cv(df, y, 10, **config)
                end = time.time()

                add = (
                    f"Tuning graphics | "
                    f"{vect} | "
                    f"{model} | "
                    f"{params}"
                )

                save_model_info(config, results, end - start, add=add)


def tuning_early_fusion(df, y):

    text_vectorizers = ["tfidf", "distilbert"]
    image_vectorizers = ["resnet18", "resnet50"]

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

    text_vectorizers = ["tfidf", "distilbert"]
    image_vectorizers = ["resnet18", "resnet50"]

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
