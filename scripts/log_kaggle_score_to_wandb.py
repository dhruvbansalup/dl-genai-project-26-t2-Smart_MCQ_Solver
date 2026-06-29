'''
To upload kaggle score to wandb

Example usage:
python -m scripts.log_kaggle_score_to_wandb --score 0.85 --id "run-id"
'''

import argparse
from src.utils.wandb_utils import log_kaggle_score_to_wandb

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--id", type=str, required=True)

    args=parser.parse_args()

    log_kaggle_score_to_wandb(kaggle_public_score=args.score, wandb_run_id=args.id)

if __name__=="__main__":
    main()