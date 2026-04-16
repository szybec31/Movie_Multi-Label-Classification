import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class LabelTransform:
    def __init__(self, df):
        self.df = df.copy()
        self.y = None

        self.y_labels = None   # nazwy klas
        self.y_count = None    # liczność klas
        self.mlb = MultiLabelBinarizer()

    def preprocessing(self):

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

        return self.y
