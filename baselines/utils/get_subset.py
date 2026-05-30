from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

def get_subset(df, y, test_size=0.2, random_state=42):

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    idx = np.arange(len(df))

    train_idx, subset_idx = next(
        msss.split(idx, y)
    )

    df_subset = df.iloc[subset_idx].reset_index(drop=True)
    y_subset = y[subset_idx]

    return df_subset, y_subset