import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bert2.data_clean import clean_data
from transformers import TFBertModel
from transformers import BertTokenizer
import csv

def bert_encode(data, maximum_len):
    input_ids = []
    attention_masks = []

    for i in range(len(data.text)):
        encoded = TOKENIZER.encode_plus(data.text[i],
                                        add_special_tokens=True,
                                        max_length=maximum_len,
                                        pad_to_max_length=True,
                                        return_attention_mask=True)

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return np.array(input_ids), np.array(attention_masks)



if __name__ == '__main__':
    file_path = "../data.csv"

    data = pd.read_csv(file_path)

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

    bert_base = TFBertModel.from_pretrained('bert-base-cased')
    TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")

    BATCH_SIZE = 8

    EPOCHS = 5

    # we will not be using metadata
    USE_META = False

    ADD_DENSE = False
    DENSE_DIM = 64

    ADD_DROPOUT = True
    DROPOUT = .2

    TRAIN_BASE = True


    def build_model(model_layer, learning_rate, use_meta=USE_META, add_dense=ADD_DENSE,
                    dense_dim=DENSE_DIM, add_dropout=ADD_DROPOUT, dropout=DROPOUT):

        input_ids = tf.keras.Input(shape=(512,), dtype='int32')
        attention_masks = tf.keras.Input(shape=(512,), dtype='int32')

        transformer_layer = model_layer([input_ids, attention_masks])

        output = transformer_layer[1]

        if add_dense:
            print("Training with additional dense layer...")
            output = tf.keras.layers.Dense(dense_dim, activation='relu')(output)

        if add_dropout:
            print("Training with dropout...")
            output = tf.keras.layers.Dropout(dropout)(output)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

        print("Training without meta-data...")
        model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

        model.compile(tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        return model


    print('Encoding Tickets...')
    print("")
    train_input_ids, train_attention_masks = bert_encode(train, 512)
    test_input_ids, test_attention_masks = bert_encode(test, 512)
    print("")
    print('Tickets encoded successfully!')

    BERT_base = build_model(bert_base, learning_rate=1e-5)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('base_model.h5', monitor='val_loss', save_best_only=True,
                                                    save_weights_only=True)
    history = BERT_base.fit([train_input_ids, train_attention_masks], train.target, validation_split=.2, epochs=EPOCHS,
                            callbacks=[checkpoint], batch_size=BATCH_SIZE)

    preds_base = BERT_base.predict([test_input_ids, test_attention_masks])

    result = pd.DataFrame()
    preds_base_1d = preds_base.flatten()
    result['prob'] = preds_base_1d
    result['predict'] = np.round(result['prob']).astype(int)
    header = ['label' , 'predict']

    with open('../val_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(header)
        writer.writerows(zip(result['prob'], result['predict']))


