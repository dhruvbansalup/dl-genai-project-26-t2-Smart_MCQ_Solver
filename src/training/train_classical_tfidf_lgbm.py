import wandb
import secrets
import argparse
import warnings
import pandas as pd
from pathlib import Path

from src.config import EnvConfig, WandbConfig
from src.utils.time_utils import time_now_ist
from src.utils.kaggle_utils import upload_model_directory_to_kagglehub
from scripts.make_processed_data_v001 import ensure_processed_data_exists

# supressing lightgbm warnings

from src.models.classical.tfidf_lgbm import TFIDFLightGBM
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def train_classical_tfidf_lgbm(log=False, uploadToKaggle=False):
    # Load processed data
    ensure_processed_data_exists()
    train_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/train_v001.parquet")

    model=TFIDFLightGBM()

    #wandb
    if log:
        from src.utils.wandb_utils import login_wandb
        login_wandb()
        model_class_name=model.__class__.__name__
        run=wandb.init(
            project=WandbConfig.PROJECT_NAME,
            entity=WandbConfig.ENTITY,
            name="Classical_TFIDF_LightGBM",
            id=f"{model_class_name}-{time_now_ist(cleaned=True)}-{secrets.token_hex(2)}",
            dir=EnvConfig.OUTPUT_DIR,
        )

        # Logging parameters to wandb
        wandb.config.update(model.config)
        wandb.config.update({
            "model_name": model.__class__.__name__,
            "pipeline": "TF-IDF + LightGBM",
        })

    print("Training TF-IDF + LightGBM model...")
    model.fit(train_df)
    print("Training completed.")

    #Save model
    MODEL_DIR=Path(EnvConfig.OUTPUT_DIR) / "classical_tfidf_lgbm"
    model.save(save_dir=MODEL_DIR)

    #Upload to Kagglehub
    if uploadToKaggle:
        upload_model_directory_to_kagglehub(local_model_dir=MODEL_DIR, variant_name="classical_tfidf_lgbm")

    if log:
        wandb.finish()


if __name__ == "__main__":
    #get cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--uploadToKaggle", type=bool, default=False)
    args = parser.parse_args()

    train_classical_tfidf_lgbm(log=args.log, uploadToKaggle=args.uploadToKaggle)