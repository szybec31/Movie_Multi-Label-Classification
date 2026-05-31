import pandas as pd
from EDA import TextEDA
from label_transform import LabelTransform
from baselines.run_cv import run_cv
from baselines.utils.save_model import save_model_info
from baselines.grid_search import run_grid_search
from baselines.freeze_models import freeze_model
from baselines.utils.test_configs import freeze_configs
import time

def main(test_type: str, test_subtype: str = "text") -> None:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 20)

    # Wczytanie danych
    df = pd.read_csv("movies.csv")

    # EDA - podstawowe informacje, usunięcie null
    eda = TextEDA(df, True)
    # eda.display_dataset_basic_info()
    df = eda.drop_na()

    # Transformacja etykiet do wektorów 1 na 18
    lt = LabelTransform(df)
    y = lt.preprocessing()
    y_label = lt.y_labels
    y_count = lt.y_count

    if test_type == "text":

        for mt1 in ["random_forest"]:

            for vect1 in ["distilbert"]:

                config = {
                    "type": "text",
                    "vectorizers": [vect1],
                    "models": [mt1],
                    "balanced_list": [True],
                    #"thresholds": [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

                }

                print(config)

                start = time.time()
                results = run_cv(df, y, 10, **config)
                end = time.time()
                save_model_info(config, results, end-start)

    elif test_type == "graphics":

        for mt2 in ["random_forest"]: #, "random_forest", "mlp"
            for vect2 in ["resnet50"]:  #, "resnet50"
                config = {
                    "type": "graphics",
                    "vectorizers": [vect2],
                    "models": [mt2],
                    "balanced_list": [True],
                    #"thresholds":[0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45]
                }

                print(config)

                start = time.time()
                results = run_cv(df, y, 10, **config)
                end = time.time()
                save_model_info(config, results, end-start)

    elif test_type == "late-fusion":

        for mt1 in ["mlp"]:                                     # "svm", "logistic", "random_forest", "mlp"
            for mt2 in ["logistic", "random_forest", "mlp"]:    # "logistic", "random_forest", "mlp"
                for vect1 in ["distilbert"]:    # "tfidf",
                    for vect2 in ["resnet50"]:  # "resnet18",
                        config = {
                            "type": "late-fusion",
                            "vectorizers": [vect1, vect2],
                            "models": [mt1, mt2],
                        }

                        print(config)

                        start = time.time()
                        results = run_cv(df, y, 10, **config)
                        end = time.time()
                        save_model_info(config, results, end-start)

    elif test_type == "early-fusion":

        for mt1 in ["logistic", "random_forest", "mlp"]:
            for vect1 in ["distilbert"]:    # "tfidf",
                for vect2 in ["resnet50"]:  # "resnet18",
                    config = {
                        "type": "early-fusion",
                        "vectorizers": [vect1, vect2],
                        "models": [mt1],
                    }

                    print(config)

                    start = time.time()
                    results = run_cv(df, y, 10, **config)
                    end = time.time()
                    save_model_info(config, results, end-start)

    elif test_type == "tuning":

        tuning_type = test_subtype
        run_grid_search(df, y, tuning_type)

    elif test_type == "freeze":

        freeze_types = ["text", "graphics", "early-fusion"]
        for f_type in freeze_types:
            configs = freeze_configs(f_type)

            # Files for frozen models are invisible in PyCharm due to its lack of support for .pkl extension, but those
            # files are indeed there if you look for them in file explorer
            for config in configs:
                freeze_model(df, y, config, mlb=lt.mlb)

    elif test_type == "info":
        # Podsumowanie informacji na temat zbioru
        eda.display_summary(y=y, y_labels=y_label, y_count=y_count)
        eda.chart_summary()
        eda.class_distribution()
        leak_df = eda.check_label_leakage(y, y_label)
        print(leak_df)

if __name__ == "__main__":
    test_type = "graphics"    # "graphics", "late-fusion", "text", "early-fusion", "tuning", "freeze" or "info"
    test_subtype = "early-fusion"   # for "tuning" test_type only; you may choose: "text", "graphics", "early-fusion" or "late-fusion"
    main(test_type, test_subtype)
