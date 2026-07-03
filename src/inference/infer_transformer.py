import torch
import argparse
import pandas as pd
from pathlib import Path

from src.models.transformers.bert import BERT
from src.models.transformers.roberta import RoBERTa
from src.datamodules.mcq_datamodule import MCQDataModule
from src.tokenizers.hf_tokenizer import HFTokenizer
from src.config import EnvConfig, GeneralConfig
from scripts.make_processed_data_v001 import ensure_processed_data_exists
from src.utils.kaggle_utils import load_kagglehub_model
from src.utils.prediction_utils import predict_top3

def infer_transformer(model_handle, checkpoint_name, model_class, tokenizer_model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Loading Model
    model=load_kagglehub_model(model_handle=model_handle, ckpt_name=checkpoint_name, model_class=model_class, load_to_device=device)

    # Get tokenized test data
    tokenizer = HFTokenizer(model_name=tokenizer_model_name)

    ensure_processed_data_exists()
    test_df=pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet")
    datamodule = MCQDataModule(
        df=None,
        test_df=test_df,
        tokenizer=tokenizer,
        max_length=GeneralConfig.MAX_LENGTH*2+2,# we have pairs, and 2 extra tokens for [QUESTION] and [OPTION],
        batch_size=8,
        num_workers=4,
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    #Get predictions
    predictions=predict_top3(model, test_loader, device)
    print("Predictions Generated Successfully!")

    #Build submission
    submission_df = pd.DataFrame({
        "ID": test_df.index + 1,
        "Prediction": predictions
    })
    submission_df.to_csv(Path(EnvConfig.SUBMISSION_DIR) / "submission.csv", index=False)
    print(f"Submission file created at: {Path(EnvConfig.SUBMISSION_DIR) / 'submission.csv'}")

if __name__ == "__main__":

    MODELS={
        "BERT":{
            "model_class": BERT,
            "tokenizer_model_name": "bert-base-uncased",
            "model_handle": "dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver/pyTorch/bert",
            "checkpoint_name": "BERT-epoch02-val_map31.0000.ckpt"
        },
        "RoBERTa":{
            "model_class": RoBERTa,
            "tokenizer_model_name": "roberta-base",
            "model_handle": "dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver/pyTorch/roberta",
            "checkpoint_name": "RoBERTa-epoch03-val_map31.0000.ckpt"
        },
    }

    #get cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    args = parser.parse_args()
    model_config = MODELS[args.model]

    infer_transformer(model_config["model_handle"], model_config["checkpoint_name"], model_config["model_class"], model_config["tokenizer_model_name"])