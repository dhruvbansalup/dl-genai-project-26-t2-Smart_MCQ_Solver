import pandas as pd
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm

from src.models.zeroshot.bart_large_mnli import BartLargeMNLI
from src.config import EnvConfig
from scripts.make_processed_data_v001 import ensure_processed_data_exists

def infer_zero_shot_bart_large_mnli():
    # Loading model
    model = BartLargeMNLI()
    print("Model Loaded Successfully!")

    # Loading test data
    ensure_processed_data_exists()
    test_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet")
    test_dataset=Dataset.from_pandas(test_df)

    # Inference
    predictions = []
    for row in tqdm(test_dataset, total=len(test_dataset), desc="Inference"):
        row = dict(row)
        prompt = row['prompt']
        candidate_labels = [row['A'], row['B'], row['C'], row['D'], row['E']]

        predictions.append(model.predict(prompt, candidate_labels))

    # Building submission
    submission_df = pd.DataFrame({
        "ID": test_df["id"],
        "Prediction": predictions
    })
    submission_df.to_csv(f"{EnvConfig.SUBMISSION_DIR}/submission.csv", index=False)


if __name__ == "__main__":
    infer_zero_shot_bart_large_mnli()