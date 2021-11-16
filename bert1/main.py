from transformers import BertTokenizer
# from utils import read_data
from model import model_setup_and_train, evaluate_model
from BugRepotsClassifier import BugRepotsClassifier
import csv
import pandas as pd
from utils import data_preprocess


if __name__ == '__main__':
    file_path = "../../data.csv"
    data = pd.read_csv(file_path)
    train_df, val_df, test_target = data_preprocess(data)

    BERT_MODEL_NAME = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # build and train model
    trainer = model_setup_and_train(tokenizer, train_df, val_df)

    print("-------Done Training-------")

    # Get the best performance model
    trained_model = BugRepotsClassifier.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        n_classes=1
    )
    trained_model.eval()
    trained_model.freeze()
    MAX_TOKEN_COUNT = 128
    # evaluate model on val_data
    labels, predictions = evaluate_model(trained_model, val_df, tokenizer, MAX_TOKEN_COUNT)
    print("-------Done Testing-------")
    with open('../val_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(labels, predictions))










