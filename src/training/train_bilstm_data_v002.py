import argparse
import pandas as pd

from scripts.make_processed_data_v002 import ensure_processed_data_exists_v002
from src.config import EnvConfig, GeneralConfig
from src.datamodules.mcq_datamodule_02 import MCQDataModule02
from src.tokenizers.vocab import build_vocabulary
from src.tokenizers.word_tokenizer import WordTokenizer
from src.models.lstm.bilstm import BiLSTM01
from src.training.trainer import train

def train_bilstm(log=False, uploadToKaggle=False):
    ensure_processed_data_exists_v002()

    # Loading dataset
    train_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/train_v002.parquet")
    val_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/val_v002.parquet")

    # Building Vocabulary
    vocab = build_vocabulary(train_df)

    # Initialize Tokenizer
    tokenizer = WordTokenizer(vocab)

    # Initialize Data Module
    data_module = MCQDataModule02(
        train_df=train_df,
        val_df=val_df,
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

    config_logs={
        "data_version":"v002",
        "pipeline":"Standard",
        "model_name":"custom-bilstm-01",
    }

    # Train the model
    train(DATA_MODULE=data_module, MODEL=model, max_epochs=12, log=log, upload_kaggle=uploadToKaggle, config_logs=config_logs)

if __name__ == "__main__":
    #get cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--uploadToKaggle", type=bool, default=False)
    args = parser.parse_args()

    train_bilstm(log=args.log, uploadToKaggle=args.uploadToKaggle)