from sklearn.model_selection import train_test_split
from data_clean import clean_data
import pandas as pd


def data_preprocess(data):
    summary_length = data['summary'].apply(lambda x: len(str(x).split(" ")))
    data["summary_token_length"] = summary_length
    description_length = data['description'].apply(lambda x: len(str(x).split(" ")))
    data["description_token_length"] = description_length

    data['summary'] = str(data['summary'])
    data['description'] = str(data['description'])
    data = clean_data(data, ['summary', 'description'])

    data['combined'] = data['summary'] + ' ' + data['description']

    positives = data[data['label'] == 1.0]
    negatives = data[data['label'] == 0.0]

    positives_train, positives_test = train_test_split(positives, test_size=0.33)
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
