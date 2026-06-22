import pandas as pd

from src.config import EnvConfig, GeneralConfig
from scripts.make_processed_data_v001 import ensure_processed_data_exists
from src.tokenizers.vocab import build_vocabulary
from src.tokenizers.word_tokenizer import WordTokenizer
from src.datamodules.mcq_datamodule import MCQDataModule
from src.models.lstm.bilstm import BiLSTM01
from src.training.trainer import train

if __name__ == "__main__":
    ensure_processed_data_exists()

    # Loading dataset
    train_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/train_v001.parquet")

    # Building Vocabulary
    vocab = build_vocabulary(train_df)

    # Initialize Tokenizer
    tokenizer = WordTokenizer(vocab)

    # Initialize Data Module
    data_module = MCQDataModule(
        df=train_df,
        test_df=None,
        tokenizer=tokenizer,
        max_length=GeneralConfig.MAX_LENGTH*2+2,# we have pairs, and 2 extra tokens for [QUESTION] and [OPTION],
        batch_size=32,
        num_workers=4,
    )

    # Initialize Model
    model = BiLSTM01(
        lr=1e-3,
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_size=256,
    )

    # attach vocab to model for encoding during inference
    model.attach_vocab(vocab)

    # Train the model
    train(DATA_MODULE=data_module, MODEL=model, max_epochs=12, log=True, upload_kaggle=True, trainer_gradient_clip_val=1.0)