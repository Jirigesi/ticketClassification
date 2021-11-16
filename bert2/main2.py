import tensorflow as tf
import pandas as pd
import numpy as np
from data_prepare import data_preprocess
from transformers import TFBertModel
from transformers import BertTokenizer
import csv
from encoder import bert_encode
from bert_model import build_model

if __name__ == '__main__':
    file_path = "../../data.csv"

    data = pd.read_csv(file_path)

    train, test, test_target = data_preprocess(data)

    bert_base = TFBertModel.from_pretrained('bert-base-cased')
    TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")

    BATCH_SIZE = 8

    EPOCHS = 1

    print('Encoding Tickets...')
    print("")
    train_input_ids, train_attention_masks = bert_encode(TOKENIZER, train, 64)
    test_input_ids, test_attention_masks = bert_encode(TOKENIZER, test, 64)
    print("")
    print('Tickets encoded successfully!')

    BERT_base = build_model(bert_base, learning_rate=1e-5)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('base_model.h5', monitor='val_loss', save_best_only=True,
                                                    save_weights_only=True)
    history = BERT_base.fit([train_input_ids, train_attention_masks], train.target, validation_split=.2, epochs=EPOCHS,
                            callbacks=[checkpoint], batch_size=BATCH_SIZE)
    print('Finish model tuning!')

    preds_base = BERT_base.predict([test_input_ids, test_attention_masks])
    preds_base_1d = preds_base.flatten()

    with open('../val_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(["label", "predict"])
        writer.writerows(zip(test_target, preds_base_1d))
    print("Finish validation!")


