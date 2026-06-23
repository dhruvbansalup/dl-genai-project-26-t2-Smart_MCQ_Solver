import os
from pathlib import Path

from dotenv import load_dotenv
from src.utils.environment import get_environment

ENV = get_environment()

load_dotenv()

class GeneralConfig:
    NUM_OPTIONS=5
    SEED=42
    # From EDA, max len of prompt or option is 118 words
    MAX_LENGTH=120

class WandbConfig:
    PROJECT_NAME="24f1001707-t22026"
    WANDB_API_KEY=os.getenv("WANDB_API_KEY")

class KaggleConfig:
    KAGGLE_USERNAME=os.getenv("KAGGLE_USERNAME")
    KAGGLE_API_TOKEN=os.getenv("KAGGLE_API_TOKEN")
    KAGGLEHUB_MODEL_REPO="dl-genai-project-26-t2-smart-mcq-solver"

class EnvConfig:
    if ENV == "kaggle":
        RAW_DATA_DIR = "/kaggle/input/competitions/smart-mcq-solver-challenge/"
        PROCESSED_DATA_DIR="/kaggle/working/data/processed"
        SUBMISSION_DIR="/kaggle/working/"
        OUTPUT_DIR="/kaggle/working/outputs"
        CHECKPOINT_DIR="/kaggle/working/outputs/checkpoints"

    elif ENV == "colab":
        RAW_DATA_DIR = "/content/data/raw"
        PROCESSED_DATA_DIR="/content/data/processed"
        SUBMISSION_DIR="/content/data/submissions"
        OUTPUT_DIR="/content/outputs"
        CHECKPOINT_DIR="/content/outputs/checkpoints"

    else: # Local
        ROOT=Path.cwd()

        RAW_DATA_DIR = ROOT / "data/raw"
        PROCESSED_DATA_DIR= ROOT / "data/processed"
        SUBMISSION_DIR= ROOT / "data/submissions"
        OUTPUT_DIR= ROOT / "outputs"
        CHECKPOINT_DIR= ROOT / "outputs/checkpoints"

#Directory Creation
Path(EnvConfig.PROCESSED_DATA_DIR).mkdir(
    parents=True,
    exist_ok=True
)
Path(EnvConfig.SUBMISSION_DIR).mkdir(
    parents=True,
    exist_ok=True
)
Path(EnvConfig.OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)
Path(EnvConfig.CHECKPOINT_DIR).mkdir(
    parents=True,
    exist_ok=True
)