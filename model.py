from BugReportsDataset import BugReportsDataset
from BugRepotsClassifier import BugRepotsClassifier
from BugReportsDataModule import BugReportsDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def model_setup_and_train(tokenizer, train_df, val_df):
    # hyperparameters
    N_EPOCHS = 5
    BATCH_SIZE = 16
    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5

    # data modules
    data_module = BugReportsDataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    model = BugRepotsClassifier(
        n_classes=1,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="bug-tickets")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)
    return trainer


def evaluate_model(trained_model, val_df, tokenizer, MAX_TOKEN_COUNT):
    val_dataset = BugReportsDataset(
        val_df,
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT
    )
    labels = []
    predictions = []
    for item in val_dataset:
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0),
            item["attention_mask"].unsqueeze(dim=0)
        )

        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())

    return labels, predictions
