import torch
import pandas as pd
from pathlib import Path

from src.models.lstm.bilstm import BiLSTM01
from src.datamodules.mcq_datamodule import MCQDataModule
from src.tokenizers.word_tokenizer import WordTokenizer
from src.config import EnvConfig, GeneralConfig
from scripts.make_processed_data_v001 import ensure_processed_data_exists
from src.utils.kaggle_utils import load_kagglehub_model
from src.utils.prediction_utils import predict_top3

def infer_bilstm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    handle = "dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver/pyTorch/bilstm01"
    checkpoint_name="BiLSTM01-epoch05-val_map30.9767-v1.ckpt"

    #Loading Model
    model=load_kagglehub_model(model_handle=handle, ckpt_name=checkpoint_name, model_class=BiLSTM01, load_to_device=device)

    # Get tokenized test data
    tokenizer = WordTokenizer(model.vocab) # type: ignore
    ensure_processed_data_exists()
    test_df=pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet")
    datamodule = MCQDataModule(
        df=None,
        test_df=test_df,
        tokenizer=tokenizer,
        max_length=GeneralConfig.MAX_LENGTH*2+2,# we have pairs, and 2 extra tokens for [QUESTION] and [OPTION],
        batch_size=32,
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
    infer_bilstm()