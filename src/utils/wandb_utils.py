import os
import wandb

def login_wandb(key=None):
    key=key or os.getenv("WANDB_API_KEY")
    if key:
        wandb.login(key=key)
    else:
        print("WANDB_API_KEY not found.")