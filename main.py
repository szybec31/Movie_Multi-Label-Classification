import pandas as pd
from My_preprocessing import My_Preprocessing_Class

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

# Wczytanie danych
df = pd.read_csv('movies.csv')

# Preprocessing
prepro = My_Preprocessing_Class(df)

prepro.display_dataset_basic_info()

X, y = prepro.preprocessing()

prepro.display_summary()

#prepro.chart_summary()
