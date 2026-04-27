import pandas as pd
from EDA import TextEDA
from label_transform import LabelTransform
import numpy as np
from baselines.run_experiment import run_experiment
from baselines.run_cv import run_cv
from baselines.utils.save_model import save_model_info

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

# Wczytanie danych
df = pd.read_csv("movies.csv")

# EDA - podstawowe informacje, usunięcie null
eda = TextEDA(df)
# eda.display_dataset_basic_info()
df = eda.drop_na()

# Transformacja etykiet do wektorów 1 na 18
lt = LabelTransform(df)
y = lt.preprocessing()
y_label = lt.y_labels
y_count = lt.y_count

# config = {
#     "type": "early-fusion", # text or graphics; soon also early-fusion and late-fusion
#     "balanced": True,
#     "vectorizers": ["tfidf", "resnet18"],   # "resnet50", "resnet18", "tfidf", "distilbert"
#     "model": "svm",
#     "max_iter_svm": 10000,
# }

# avg, std = run_cv(df, y, 5, **config)

# print(config)
# for k in avg:
#     print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

# save_model_info(config, avg, std, "test")

# exit()

for mt1 in ["svm", "logistic", "random_forest", "mlp"]:
    for mt2 in ["logistic", "random_forest", "mlp"]:

        if mt1 == "svm" and mt2 == "svm":
            continue

        for vect1 in ["tfidf", "distilbert"]:
            for vect2 in ["resnet18", "resnet50"]:
                config = {
                    "type": "late-fusion",
                    "balanced_list": [True, True],
                    "vectorizers": [vect1, vect2],
                    "models": [mt1, mt2],
                    "max_features_tfidf": 20000,
                    "max_iter": 20,
                    "learning_rate_init": 0.001,
                    "max_depth": 5,
                    "max_iter_svm": 5000,
                }

                print(config)

                avg_list, std_list = run_cv(df, y, 5, **config)

                names = config["models"] + ["late-fusion-or", "late-fusion-and", "late-fusion-avg"]

                for i, (avg, std) in enumerate(zip(avg_list, std_list)):
                    print(f"\n=== {names[i]} ===")

                    for k in avg:
                        print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")
                
                save_model_info(config, avg_list, std_list, "test")

                

exit()

# Podsumowanie informacji na temat zbioru
eda.display_summary(y=y,y_labels=y_label,y_count=y_count)
# eda.chart_summary()