import pandas as pd
from sklearn.model_selection import train_test_split
import re
from data_clean import clean_data
# def read_data(file_path, RANDOM_SEED=42):
#     df = pd.read_csv(file_path)
#     df = clean(df, )
#
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
#     train_df.shape, val_df.shape
#
#     not_bug_train_df = train_df[train_df['label'] == 1]
#     bug_train_df = train_df[train_df['label'] == 0]
#     not_bug_train_df.shape, bug_train_df.shape
#
#     train_df = pd.concat([
#         not_bug_train_df,
#         bug_train_df.sample(6400)
#     ])
#     train_df = train_df.sample(frac=1)
#     val_df = val_df.sample(frac=1)
#
#     train_df = train_df.reset_index(drop=True)
#     val_df = val_df.reset_index(drop=True)
#
#     return train_df, val_df



def data_preprocess(data):
    summary_length = data['summary'].apply(lambda x: len(str(x).split(" ")))
    data["summary_token_length"] = summary_length
    description_length = data['description'].apply(lambda x: len(str(x).split(" ")))
    data["description_token_length"] = description_length

    data = clean_data(data, ['summary', 'description'])

    data['combined'] = data['summary'] + ' ' + data['description']

    positives = data[data['label'] == 1.0]
    negatives = data[data['label'] == 0.0]

    positives_train, positives_test = train_test_split(positives, test_size=0.2)
    positive_train_nums = positives_train.shape[0]

    # get 1: 1 negative training data
    negative_total_nums = data[data['label'] == 0.0].shape[0]
    negative_test_size = 1 - positive_train_nums / negative_total_nums
    negatives_train, negatives_test = train_test_split(negatives, test_size=negative_test_size)

    train = pd.concat([positives_train, negatives_train]).sample(frac=1).reset_index(drop=True)
    test = pd.concat([positives_test, negatives_test]).sample(frac=1).reset_index(drop=True)

    # cadidates: summary, description, combined
    SELECT_TEXT = "combined"
    train = train[[SELECT_TEXT, "label"]]
    test_target = test['label']
    test = test[[SELECT_TEXT]]
    train = train.rename(columns={SELECT_TEXT: "text", "label": "target"})
    test = test.rename(columns={SELECT_TEXT: "text"})

    return train, test, test_target

