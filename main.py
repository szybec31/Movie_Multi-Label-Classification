import pandas as pd
from EDA import TextEDA
from label_transform import LabelTransform
import numpy as np
from add_posters import attach_posters
from baselines.run_model import run_model

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

# Wczytanie danych
df = pd.read_csv("movies.csv")
df, status = attach_posters(df)
if status:
    df.to_csv('movies.csv', index=False)

# EDA - podstawowe informacje, usunięcie null
eda = TextEDA(df)
# eda.display_dataset_basic_info()
df = eda.drop_na()

# Transformacja etykiet do wektorów 1 na 18
lt = LabelTransform(df)
y = lt.preprocessing()
y_label = lt.y_labels
y_count = lt.y_count

print(20*"=")
print("text balanced 0.5")
run_model("tfidf", df, y, y_label, ["text", 0.5, True])

print(20*"=")
print("text imbalanced 0.2")
run_model("tfidf", df, y, y_label, ["text", 0.2, False])

# print(20*"=")
# print("title")
# run_model("tfidf", df, y, y_label, "title")

# print(20*"=")
# print("overview")
# run_model("tfidf", df, y, y_label, "overview")

exit()

# Podsumowanie informacji na temat zbioru
eda.display_summary(y=y,y_labels=y_label,y_count=y_count)
# eda.chart_summary()