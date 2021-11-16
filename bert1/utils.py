import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(file_path, RANDOM_SEED=42):
    df = pd.read_csv(file_path)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    train_df.shape, val_df.shape

    not_bug_train_df = train_df[train_df['label'] == 1]
    bug_train_df = train_df[train_df['label'] == 0]
    not_bug_train_df.shape, bug_train_df.shape

    train_df = pd.concat([
        not_bug_train_df,
        bug_train_df.sample(6400)
    ])
    train_df = train_df.sample(frac=1)
    val_df = val_df.sample(frac=1)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


