import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from utils import read_data
from model import model_setup_and_train, evaluate_model
from BugRepotsClassifier import BugRepotsClassifier
import csv


if __name__ == '__main__':
    file_path = "../data.csv"
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    train_df, val_df = read_data(file_path, RANDOM_SEED)
    BERT_MODEL_NAME = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    # train model
    trainer = model_setup_and_train(tokenizer, train_df, val_df)

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

    with open('val_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(labels, predictions))










