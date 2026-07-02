import argparse
import pandas as pd

from src.config import EnvConfig, GeneralConfig
from scripts.make_processed_data_v001 import ensure_processed_data_exists
from src.tokenizers.hf_tokenizer import HFTokenizer
from src.datamodules.mcq_datamodule import MCQDataModule
from src.training.trainer import train
from src.models.transformers.bert import BERT
from src.models.transformers.roberta import RoBERTa

def train_transformer(model_class, tokenizer_model_name, log=False, uploadToKaggle=False):
    ensure_processed_data_exists()

    # Loading dataset
    train_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/train_v001.parquet")

    # Initialize Tokenizer
    tokenizer = HFTokenizer(model_name=tokenizer_model_name)

    # Initialize Data Module
    data_module = MCQDataModule(
        df=train_df,
        test_df=None,
        tokenizer=tokenizer,
        max_length=GeneralConfig.MAX_LENGTH*2+2,# we have pairs, and 2 extra tokens for [QUESTION] and [OPTION],
        batch_size=8,
        num_workers=4,
    )

    # Initialize Model
    model = model_class(lr=2e-5)

    # Train the model
    train(DATA_MODULE=data_module, MODEL=model, max_epochs=6, log=log, upload_kaggle=uploadToKaggle)

if __name__ == "__main__":

    MODELS={
        # model_name: (model_class, tokenizer_model_name)
        "BERT":(BERT, "bert-base-uncased"),
        "RoBERTa":(RoBERTa, "roberta-base"),
    }

    #get cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--uploadToKaggle", type=bool, default=False)
    args = parser.parse_args()

    model_class, tokenizer_model_name=MODELS[args.model]

    train_transformer(model_class=model_class, tokenizer_model_name=tokenizer_model_name, log=args.log, uploadToKaggle=args.uploadToKaggle)