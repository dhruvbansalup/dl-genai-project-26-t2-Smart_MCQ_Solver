
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgbm
from pathlib import Path

from src.config import EnvConfig

def load_processed_data(version="v001"):
    train_df = pd.read_parquet(Path(EnvConfig.PROCESSED_DATA_DIR) / f"train_{version}.parquet")
    test_df = pd.read_parquet(Path(EnvConfig.PROCESSED_DATA_DIR) / f"test_{version}.parquet")
    return train_df, test_df

def preprocessing(df):
    #Combine prompt and options into a single text column
    for opt in ["A","B","C","D","E"]:
        df["prompt"] += "[SEP]" +opt+": " + df[opt].fillna("")
    
    if "answer" in df.columns:
        return df[["prompt","answer"]]
    else:
        return df[["prompt"]]

def tfidf_lgbm():
    train_df, test_df = load_processed_data("v001")

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    #Encode labels
    le= LabelEncoder()
    train_df["answer"] = pd.Series(le.fit_transform(train_df["answer"]))
    
    #TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=5, max_df=0.9)
    
    X_train = tfidf.fit_transform(train_df["prompt"])
    X_test = tfidf.transform(test_df["prompt"])
    
    print("TF-IDF Vectorization Completed")
    
    #Model
    model = lgbm.LGBMClassifier(n_estimators=100, learning_rate=0.1)

    # supressing lightgbm warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    #Train LightGBM
    y_train = train_df["answer"]
    model.fit(X_train, y_train)

    print("Model Training Completed")

    #Get Probabilities
    probabilities = np.asanyarray(model.predict_proba(X_test))

    # converting to labels top3
    labels = le.classes_
    predictions = []
    for p in probabilities:
        top3_indices = np.argsort(p)[-3:][::-1]
        predictions.append(" ".join(labels[top3_indices]))

    #Submission File
    submission_df = pd.DataFrame({
        "ID": test_df.index+1,
        "Prediction": predictions
    })

    submission_df.to_csv(Path(EnvConfig.SUBMISSION_DIR) / "submission.csv", index=False)

    print("Submission File Created")


if __name__ == "__main__":
    tfidf_lgbm()