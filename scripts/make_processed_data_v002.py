import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing.formatting import format_mcq_data
from src.config import EnvConfig, GeneralConfig

def split_train_validation(df):
    """
    Spliting the training data into training and validation sets based on semantic similarity of prompts.
    """

    model=SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(
        df['prompt'].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Using cosine similarity to find similar prompts
    similarity_matrix = cosine_similarity(embeddings)

    # Using Union-Find to group similar prompts
    parent = list(range(len(df)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    #Merging similar prompts based on a threshold
    THRESHOLD = 0.95
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i][j] >= THRESHOLD:
                union(i, j)

    #Assigning group labels based on the union-find structure
    groups = [find(i) for i in range(len(df))]

    # Using StratifiedGroupKFold to split the data while maintaining the distribution of groups
    sgkf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=GeneralConfig.SEED
    )
    train_idx, val_idx=next(
        sgkf.split(df, df['answer'], groups)
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    return train_df, val_df


def create_processed_data_v002():
    train_raw_path = f"{EnvConfig.RAW_DATA_DIR}/train.csv"
    test_raw_path = f"{EnvConfig.RAW_DATA_DIR}/test.csv"

    train_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/train_v002.parquet"
    validation_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/val_v002.parquet"
    test_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/test_v002.parquet"

    # Remove any existing processed files to avoid confusion
    import os
    if os.path.exists(train_processed_path):
        os.remove(train_processed_path)
    if os.path.exists(validation_processed_path):
        os.remove(validation_processed_path)
    if os.path.exists(test_processed_path):
        os.remove(test_processed_path)


    df= pd.read_csv(train_raw_path)
    df = format_mcq_data(df)

    test_df = pd.read_csv(test_raw_path)
    test_df = format_mcq_data(test_df)

    #removing exact duplicates (with shuffled options)
    df=df.drop_duplicates(subset=["prompt", "A", "B", "C", "D", "E", "answer"]).reset_index(drop=True)

    # Split the training data into training and validation sets based on sementic similarity of the prompts.
    train_df, val_df = split_train_validation(df)

    train_df.to_parquet(train_processed_path, index=False)
    val_df.to_parquet(validation_processed_path, index=False)
    test_df.to_parquet(test_processed_path, index=False)
    print("Processed data created successfully.")

def ensure_processed_data_exists_v002():
    train_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/train_v002.parquet"
    validation_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/val_v002.parquet"
    test_processed_path = f"{EnvConfig.PROCESSED_DATA_DIR}/test_v002.parquet"

    if not os.path.exists(train_processed_path) or not os.path.exists(validation_processed_path) or not os.path.exists(test_processed_path):
        print("Processed data not found. Creating processed data...")
        create_processed_data_v002()
    else:
        print("Processed data already exists.")


if __name__ == "__main__":
    create_processed_data_v002()
