
import pandas as pd
from pathlib import Path

from src.config import EnvConfig
from src.models.classical.tfidf_lgbm import TFIDFLightGBM
from src.utils.kaggle_utils import download_model_directory_from_kagglehub
from scripts.make_processed_data_v001 import ensure_processed_data_exists


MODEL_HANDLE="dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver/pytorch/classical_tfidf_lgbm"

def infer_classical_tfidf_lgbm():

    #Download model from Kagglehub
    model_dir=download_model_directory_from_kagglehub(model_handle=MODEL_HANDLE)
    if model_dir is None:
        raise RuntimeError(
            "Failed to download model from KaggleHub."
        )

    #Loading Model
    model=TFIDFLightGBM.load(model_dir=model_dir)

    #Loading Data
    ensure_processed_data_exists()
    test_df=pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet")

    #Get predictions
    predictions=model.predict_top3(test_df)

    # Build submission
    submission_df = pd.DataFrame({
        "ID": test_df.index + 1,
        "Prediction": predictions
    })
    submission_df.to_csv(Path(EnvConfig.SUBMISSION_DIR) / "submission.csv", index=False)
    print(f"Submission file created at: {Path(EnvConfig.SUBMISSION_DIR) / 'submission.csv'}")

if __name__ == "__main__":
    infer_classical_tfidf_lgbm()