import pandas as pd
from EDA import TextEDA
from label_transform import LabelTransform
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

# Wczytanie danych
df = pd.read_csv("movies.csv")

# EDA - podstawowe informacje, usunięcie null
eda = TextEDA(df)
eda.display_dataset_basic_info()
df = eda.drop_na()


# Transformacja etykiet do wektorów 1 na 18
lt = LabelTransform(df)
y = lt.preprocessing()
y_label = lt.y_labels
y_count = lt.y_count

# Podsumowanie informacji na temat zbioru
eda.display_summary(y=y,y_labels=y_label,y_count=y_count)
eda.chart_summary()