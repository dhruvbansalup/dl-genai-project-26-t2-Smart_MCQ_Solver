
import secrets

import wandb
from src.utils.wandb_utils import login_wandb
from src.config import EnvConfig, WandbConfig
from src.utils.time_utils import time_now_ist

def log_weighted_ensemble():
    name = "WeightedEnsemble-BERT-DeBERTa-RoBERTa"

    login_wandb()

    run=wandb.init(
        project=WandbConfig.PROJECT_NAME,
        entity=WandbConfig.ENTITY,
        name=name,
        id=f"{name}-{time_now_ist(cleaned=True)}-{secrets.token_hex(2)}",
        dir=EnvConfig.OUTPUT_DIR,
    )
    wandb.config.update({
        "model_name": "Weighted Ensemble of BERT, DeBERTa, RoBERTa",
        "pipeline": "Weighted Ensemble",

        "weights": {
            "bert-base-uncased": 0.4,
            "microsoft/deberta-v3-base": 0.4,
            "roberta-base": 0.2
        },
        "trained": False
    })

    wandb.summary["kaggle_public_score"] = 0.74106

    wandb.finish()

def log_zero_shot_BartLargeMNLI():
    name="ZeroShot-BartLargeMNLI"

    login_wandb()
    run=wandb.init(
        project=WandbConfig.PROJECT_NAME,
        entity=WandbConfig.ENTITY,
        name=name,
        id=f"{name}-{time_now_ist(cleaned=True)}-{secrets.token_hex(2)}",
        dir=EnvConfig.OUTPUT_DIR,
    )

    wandb.config.update({
        "pipeline": "Zero Shot",
        "model_name": "facebook/bart-large-mnli",
        "trained": False
    })

    wandb.summary["kaggle_public_score"] = 0.49958

    wandb.finish()

def log_sample_submission():
    name="SampleSubmission"

    login_wandb()
    run=wandb.init(
        project=WandbConfig.PROJECT_NAME,
        entity=WandbConfig.ENTITY,
        name=name,
        id=f"{name}-{time_now_ist(cleaned=True)}-{secrets.token_hex(2)}",
        dir=EnvConfig.OUTPUT_DIR,
    )

    wandb.config.update({
        "pipeline": "Sample Submission",
        "trained": False
    })

    wandb.summary["kaggle_public_score"] = 0.35245

    wandb.finish()


if __name__ == "__main__":
    # log_weighted_ensemble()
    # log_zero_shot_BartLargeMNLI()
    # log_sample_submission()
    pass