import wandb
from src.config import WandbConfig

def login_wandb(key=None):
    key=key or WandbConfig.WANDB_API_KEY
    if key:
        wandb.login(key=key)
    else:
        print("WANDB_API_KEY not found.")


def log_kaggle_score_to_wandb(kaggle_public_score, wandb_run_id):
    # get previous run using the run_id
    api = wandb.Api()
    run = api.run(f"{WandbConfig.ENTITY}/{WandbConfig.PROJECT_NAME}/{wandb_run_id}")

    run.summary["kaggle_public_score"] = kaggle_public_score

    run.summary.update()
