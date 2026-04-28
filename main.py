import pandas as pd
from EDA import TextEDA
from label_transform import LabelTransform
import numpy as np
from baselines.run_experiment import run_experiment
from baselines.run_cv import run_cv
from baselines.utils.save_model import save_model_info
import time

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

test_type = "text" # or "graphics" or "late-fusion"

if test_type == "text":

    for mt1 in ["svm", "logistic", "random_forest", "mlp"]:

        for vect1 in ["tfidf", "distilbert"]:

            config = {
                "type": "text",
                "balanced": True,
                "vectorizer": vect1,
                "model": mt1,
                "max_features_tfidf": 20000,
                "max_iter": 40,
                "learning_rate_init": 0.001,
                "max_depth": 5,
                "max_iter_svm": 5000,
            }

            print(config)

            start = time.time()

            avg_list, std_list = run_cv(df, y, 5, **config)

            end = time.time()

            print(f"Time: {(end - start)}")

            names = [config["model"]] + ["late-fusion-or", "late-fusion-and", "late-fusion-avg"]

            for i, (avg, std) in enumerate(zip(avg_list, std_list)):
                print(f"\n=== {names[i]} ===")

                for k in avg:
                    print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
            
            save_model_info(config, avg_list, std_list, "test\\text", (end - start))

elif test_type == "graphics":

    for mt2 in ["logistic", "random_forest", "mlp"]:
        for vect2 in ["resnet18", "resnet50"]:
            config = {
                "type": "graphics",
                "balanced": True,
                "vectorizer": vect2,
                "model": mt2,
                "max_iter": 40,
                "learning_rate_init": 0.001,
                "max_depth": 5,
                "max_iter_svm": 5000,
            }

            print(config)

            start = time.time()

            avg_list, std_list = run_cv(df, y, 5, **config)

            end = time.time()

            print(f"Time: {(end - start)}")

            names = [config["model"]]+ ["late-fusion-or", "late-fusion-and", "late-fusion-avg"]

            for i, (avg, std) in enumerate(zip(avg_list, std_list)):
                print(f"\n=== {names[i]} ===")

                for k in avg:
                    print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
            
            save_model_info(config, avg_list, std_list, "test\\graphics", (end - start))

elif test_type == "late-fusion":

    for mt1 in ["svm", "logistic", "random_forest", "mlp"]:
        for mt2 in ["logistic", "random_forest", "mlp"]:

            if mt1 == "svm" and (mt2 == "logistic" or  mt2 == "random_forest"):
                continue

            for vect1 in ["tfidf", "distilbert"]:
                for vect2 in ["resnet18", "resnet50"]:
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

                    avg_list, std_list = run_cv(df, y, 5, **config)

                    end = time.time()

                    print(f"Time: {(end - start)}")

                    names = config["models"] + ["late-fusion-or", "late-fusion-and", "late-fusion-avg"]

                    for i, (avg, std) in enumerate(zip(avg_list, std_list)):
                        print(f"\n=== {names[i]} ===")

                        for k in avg:
                            print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
                    
                    save_model_info(config, avg_list, std_list, "test\\late-fusion", (end - start), False)

                

exit()

# Podsumowanie informacji na temat zbioru
eda.display_summary(y=y,y_labels=y_label,y_count=y_count)
eda.chart_summary()
eda.class_distribution()