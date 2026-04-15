import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class My_Preprocessing_Class:
    def __init__(self, df):
        self.df = df.copy()

        self.X = None
        self.y = None

        self.y_labels = None   # nazwy klas
        self.y_count = None    # liczność klas
        self.mlb = MultiLabelBinarizer()

    def display_dataset_basic_info(self):
        print("Podstawowe informacje o zbiorze danych")
        print(50 * "==")
        print(self.df.head(3))
        print(50 * "==")
        print(self.df.info())
        print(50 * "==")

    def preprocessing(self):
        # Usunięcie brakujących danych
        self.df = self.df.dropna(subset=['overview'])

        # Bezpieczna konwersja genre -> lista
        def safe_eval(x):
            if pd.isnull(x):
                return []
            if str(x).lower() == 'nan':
                return []
            try:
                return ast.literal_eval(x)
            except:
                return []

        self.df['genre'] = self.df['genre'].apply(safe_eval)

        # MultiLabelBinarizer
        self.y = self.mlb.fit_transform(self.df['genre'])
        self.y_labels = self.mlb.classes_

        # Liczność klas
        self.y_count = np.sum(self.y, axis=0)

        # Przygotowanie tekstu
        self.df['text'] = self.df['title'] + " " + self.df['overview']
        self.X = self.df['text']

        return self.X, self.y

    def display_summary(self):
        # długości tekstów
        self.df['title_length'] = self.df['title'].apply(len)
        self.df['overview_length'] = self.df['overview'].apply(len)

        print('ilość obserwacji (n_samples): ', self.y.shape[0])
        print('ilość cech (n_features): ', 1)  # mamy jeden tekst (połączony)
        print('ilość unikalnych etykiet (n_labels): ', self.y.shape[1])

        print("W ilości znaków...")
        print('długość tytułu max = ', self.df['title_length'].max())
        print('długość opisu max = ', self.df['overview_length'].max())
        print(f"średnia długość tytułu = {self.df['title_length'].mean():.2f}")
        print(f"średnia długość opisu = {self.df['overview_length'].mean():.2f}")

        print(50 * "==")

        summary = pd.DataFrame({
            "Kategoria": self.y_labels,
            "Ilość": self.y_count
        })

        print(summary.sort_values(by="Ilość", ascending=False))

    def chart_summary(self):

        self.df['title_words'] = self.df['title'].apply(lambda x: len(x.split()))
        self.df['overview_words'] = self.df['overview'].apply(lambda x: len(x.split()))

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(self.df['title_words'], bins=30)
        plt.title('Histogram liczby słów w tytułach')
        plt.xlabel('Liczba słów')
        plt.ylabel('Liczba filmów')
        plt.grid()

        plt.subplot(1,2,2)
        plt.hist(self.df['overview_words'], bins=30)
        plt.title('Histogram liczby słów w opisach')
        plt.xlabel('Liczba słów')
        plt.ylabel('Liczba filmów')
        plt.grid()
        plt.show()