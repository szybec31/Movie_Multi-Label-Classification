import pandas as pd
from EDA import TextEDA
from label_transform import LabelTransform
import numpy as np
from baselines.run_experiment import run_experiment
from baselines.run_cv import run_cv
from baselines.utils.save_model import save_model_info
import time

def main(test_type: str) -> None:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 20)

    # Wczytanie danych
    df = pd.read_csv("movies.csv")

    # EDA - podstawowe informacje, usunięcie null
    eda = TextEDA(df,True)
    # eda.display_dataset_basic_info()
    df = eda.drop_na()

    # Transformacja etykiet do wektorów 1 na 18
    lt = LabelTransform(df)
    y = lt.preprocessing()
    y_label = lt.y_labels
    y_count = lt.y_count

    if test_type == "text":

        for mt1 in ["random_forest", "mlp"]:

            for vect1 in ["distilbert"]:

                config = {
                    "type": "text",
                    "balanced": True,
                    "vectorizers": [vect1],
                    "models": [mt1],
                    "max_features_tfidf": 20000,
                    "max_iter": 40,
                    "learning_rate_init": 0.001,
                    "max_depth": 5,
                    "max_iter_svm": 5000,
                }

                print(config)

                start = time.time()
                results = run_cv(df, y, 10, **config)
                end = time.time()
                save_model_info(config, results, end-start)

    elif test_type == "graphics":

        for mt2 in ["logistic", "random_forest", "mlp"]:
            for vect2 in ["resnet50"]:
                config = {
                    "type": "graphics",
                    "balanced": True,
                    "vectorizers": [vect2],
                    "models": [mt2],
                    "max_iter": 40,
                    "learning_rate_init": 0.001,
                    "max_depth": 5,
                    "max_iter_svm": 5000,
                }

                print(config)

                start = time.time()
                results = run_cv(df, y, 10, **config)
                end = time.time()
                save_model_info(config, results, end-start)

    elif test_type == "late-fusion":

        for mt1 in ["svm", "logistic", "random_forest", "mlp"]: # "svm", "logistic", "random_forest", "mlp"
            for mt2 in ["logistic", "random_forest", "mlp"]: # "logistic", "random_forest", "mlp"
                for vect1 in ["distilbert"]: #"tfidf",
                    for vect2 in ["resnet50"]: # "resnet18",
                        config = {
                            "type": "late-fusion",
                            "balanced_list": [True, True],
                            "vectorizers": [vect1, vect2],
                            "models": [mt1, mt2],
                            "max_features_tfidf": 20000,
                            "max_iter": 40,
                            "learning_rate_init": 0.001,
                            "max_depth": 5,
                            "max_iter_svm": 5000,
                        }

                        print(config)

                        start = time.time()
                        results = run_cv(df, y, 10, **config)
                        end = time.time()
                        save_model_info(config, results, end-start)

    elif test_type == "early-fusion":

        for mt1 in ["logistic", "random_forest", "mlp"]:
            for vect1 in ["distilbert"]: #"tfidf",
                for vect2 in ["resnet50"]: # "resnet18",
                    config = {
                        "type": "early-fusion",
                        "balanced": True,
                        "vectorizers": [vect1, vect2],
                        "models": [mt1],
                        "max_iter": 40,
                        "learning_rate_init": 0.001,
                        "max_depth": 5,
                        "max_iter_svm": 5000,
                    }

                    print(config)

                    start = time.time()
                    results = run_cv(df, y, 10, **config)
                    end = time.time()
                    save_model_info(config, results, end-start)

    elif test_type == "info":
        # Podsumowanie informacji na temat zbioru
        eda.display_summary(y=y,y_labels=y_label,y_count=y_count)
        eda.chart_summary()
        eda.class_distribution()
        leak_df = eda.check_label_leakage(y, y_label)
        print(leak_df)

if __name__ == "__main__":
    test_type = "graphics" #  "graphics", "late-fusion", "text", "early-fusion" or "info"
    main(test_type)