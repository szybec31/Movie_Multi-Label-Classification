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

config = {
    "type": "text", # text or graphics; soon also early-fusion and late-fusion
    "balanced": True,
    "vectorizer": "tfidf",   # "resnet50", "resnet18", "tfidf", "distilbert"
    "model": "logistic",
    "max_features_tfidf": 20000,
}

avg, std = run_cv(df, y, 5, **config)

print(config)
for k in avg:
    print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

save_model_info(config, avg, std, "test")

exit()

for model_type in ["mlp"]: # , "svm" , "logistic", "random_forest", "mlp",
    for subtype in ["text"]: # "text", "title", "overview", "graphics"

        balances = [None] #[True, False]
        
        if model_type == "logistic":
            thresholds = [0.5, 0.3, 0.2]
        else:
            thresholds = [None]

        for tr in thresholds:
            for b in balances:

                config = {
                    "type": "text", # text or graphics; soon also early-fusion and late-fusion
                    "subtype": subtype,
                    "threshold": tr,
                    "balanced": b,
                    "vectorizer": "tfidf",   # "resnet50", "resnet18", "tfidf", "distilbert"
                    "model": model_type,
                    "max_features_tfidf": 20000,
                    "max_iter": 20,
                    "learning_rate_init": 0.001
                }

                avg, std = run_cv(df, y, 5, **config)

                print(config)
                for k in avg:
                    print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

                save_model_info(config, avg, std, "test")

exit()

# Podsumowanie informacji na temat zbioru
eda.display_summary(y=y,y_labels=y_label,y_count=y_count)
# eda.chart_summary()