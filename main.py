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

'''
    config = {
        "type": "graphics", # text or graphics; soon also early-fusion and late-fusion
        "subtype": "graphics", # for text: text or title or overview
        "threshold": 0.5, # Number for logistic or None for svm
        "balanced": False, # True or False
        "vectorizer": "resnet18", # tfidf, bert, resnet18, resnet50
        "model": "logistic" # logistic, svm, rf, bert
    }

    avg, std = run_cv(df, y, config)

    exit()
'''

for model_type in ["logistic"]: # , "svm"
    for subtype in ["graphics"]: # "text", "title", "overview"

        balances = [True, False]
        
        if model_type == "logistic":
            thresholds = [0.5, 0.3, 0.2]
        else:
            thresholds = [None]

        for tr in thresholds:
            for b in balances:

                config = {
                    "type": "graphics", # text or graphics; soon also early-fusion and late-fusion
                    "subtype": subtype,
                    "threshold": tr,
                    "balanced": b,
                    "vectorizer": "resnet50",
                    "model": model_type
                }

                avg, std = run_cv(df, y, config)

                print(config)
                for k in avg:
                    print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

                save_model_info(config, avg, std, "test")

exit()

# Podsumowanie informacji na temat zbioru
eda.display_summary(y=y,y_labels=y_label,y_count=y_count)
# eda.chart_summary()