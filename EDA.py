import matplotlib.pyplot as plt
import pandas as pd

class TextEDA:
    def __init__(self, df,show=False):
        self.df = df.copy()
        self.show = show
    def drop_na(self):
        self.df = self.df.dropna(subset=['overview'])
        return self.df

    def display_dataset_basic_info(self):
        print("Podstawowe informacje o zbiorze danych")
        print(50 * "==")
        print(self.df.head(3))
        print(50 * "==")
        print(self.df.info())
        print(50 * "==")
        print("Czy jest null?")
        print(self.df.isnull().sum())
        print(50 * "==")

    def display_summary(self, y=None, y_labels=None, y_count=None):
        # długość w znakach
        self.df['title_length'] = self.df['title'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        self.df['overview_length'] = self.df['overview'].apply(lambda x: len(x) if isinstance(x, str) else 0)

        # długość w słowach
        self.df['title_words'] = self.df['title'].apply(lambda x: len(str(x).split()))
        self.df['overview_words'] = self.df['overview'].apply(lambda x: len(str(x).split()))

        print('ilość obserwacji:', len(self.df))

        if y is not None:
            print('ilość etykiet:', y.shape[1])

        print("\n--- ZNAKI ---")
        print("max title:", self.df['title_length'].max())
        print("avg title:", round(self.df['title_length'].mean(),3))
        print("max overview:", self.df['overview_length'].max())
        print("avg overview:", round(self.df['overview_length'].mean(),3))

        print("\n--- SŁOWA ---")
        print("max title:", self.df['title_words'].max())
        print("avg title:", round(self.df['title_words'].mean(),3))
        print("max overview:", self.df['overview_words'].max())
        print("avg overview:", round(self.df['overview_words'].mean(),3))

        print("\n--- PLAKATY ---")
        print("Ile filmów ma poster:", self.df["poster_path"].notnull().sum())
        coverage = self.df["poster_path"].notnull().mean()
        print("Pokrycie obrazów:", coverage)

        print(50 * "==")

        if y_labels is not None and y_count is not None:
            self.summary = pd.DataFrame({
                "Kategoria": y_labels,
                "Ilość": y_count
            })
            print(self.summary.sort_values(by="Ilość", ascending=False))

    def chart_summary(self):
        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(self.df['title_words'], bins=30)
        plt.title('Histogram liczby słów w tytułach')
        plt.xlabel('Liczba słów')
        plt.ylabel('Liczba filmów')
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.hist(self.df['overview_words'], bins=30)
        plt.title('Histogram liczby słów w opisach')
        plt.xlabel('Liczba słów')
        plt.ylabel('Liczba filmów')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("histogram.png")
        if self.show:
            plt.show()

    def class_distribution(self):
        plt.bar(self.summary['Kategoria'],self.summary['Ilość'])
        plt.grid(True)
        plt.xticks(rotation=35)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig("class_distribution.png")
        if self.show:
            plt.show()