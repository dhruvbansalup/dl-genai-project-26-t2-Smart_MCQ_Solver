import wandb
from src.config import WandbConfig

def login_wandb(key=None):
    key=key or WandbConfig.WANDB_API_KEY
    if key:
        wandb.login(key=key)
    else:
        print("WANDB_API_KEY not found.")