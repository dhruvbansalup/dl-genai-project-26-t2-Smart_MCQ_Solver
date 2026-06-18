from pathlib import Path
from src.utils.environment import get_environment

ENV = get_environment()

class EnvConfig:

    if ENV == "kaggle":
        RAW_DATA_DIR = "/kaggle/input/competitions/smart-mcq-solver-challenge/"
        PROCESSED_DATA_DIR="/kaggle/working/data/processed"
        SUBMISSION_DIR="/kaggle/working/"
    
    elif ENV == "colab":
        RAW_DATA_DIR = "/content/data/raw"
        PROCESSED_DATA_DIR="/content/data/processed"
        SUBMISSION_DIR="/content/data/submissions"
    
    else: # Local
        ROOT=Path.cwd()

        RAW_DATA_DIR = ROOT / "data/raw"
        PROCESSED_DATA_DIR= ROOT / "data/processed"
        SUBMISSION_DIR= ROOT / "data/submissions"

#Directory Creation
Path(EnvConfig.PROCESSED_DATA_DIR).mkdir(
    parents=True,
    exist_ok=True
)
Path(EnvConfig.SUBMISSION_DIR).mkdir(
    parents=True, 
    exist_ok=True
)