# Create processed data from raw csv files by cleaning text
import os
import pandas as pd

from src.preprocessing.formatting import format_mcq_data
from src.config import EnvConfig

def create_processed_data(raw_csv_path, processed_parquet_path):

    df = pd.read_csv(raw_csv_path)
    df = format_mcq_data(df)
    df.to_parquet(processed_parquet_path, index=False)

def main():
    train_raw_path = f"{EnvConfig.RAW_DATA_DIR}/train.csv"
    test_raw_path = f"{EnvConfig.RAW_DATA_DIR}/test.csv"
    train_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/train_v001.parquet"
    test_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet"

    # Remove any existing processed files to avoid confusion
    import os
    if os.path.exists(train_processed_path):
        os.remove(train_processed_path)
    if os.path.exists(test_processed_path):
        os.remove(test_processed_path)

    create_processed_data(train_raw_path, train_processed_path)
    create_processed_data(test_raw_path, test_processed_path)

    print("Processed data created successfully.")

def ensure_processed_data_exists_v001():
    train_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/train_v001.parquet"
    test_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet"

    if not os.path.exists(train_processed_path) or not os.path.exists(test_processed_path):
        print("Processed data not found. Creating processed data...")
        main()
    else:
        print("Processed data already exists.")

if __name__ == "__main__":
    main()
