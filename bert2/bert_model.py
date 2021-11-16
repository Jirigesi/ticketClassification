import tensorflow as tf

# TRAIN_BASE = True

def build_model(model_layer, learning_rate, use_meta=False, add_dense=True,
                dense_dim=64, add_dropout=True, dropout=0.2):

    input_ids = tf.keras.Input(shape=(64,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(64,), dtype='int32')

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
