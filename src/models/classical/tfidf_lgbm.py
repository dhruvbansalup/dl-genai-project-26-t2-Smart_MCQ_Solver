import json
import joblib
import numpy as np
import lightgbm as lgbm
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class TFIDFLightGBM:
    """
    Baseline model TF-IDF + LightGBM
    """
    def __init__(self,
        lr=0.1,
        n_estimators=100,
        max_features=30000,
        ngram_range=(1,2),
        min_df=5,
        max_df=0.9,
    ):
        self.lr = lr
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.LabelEncoder = LabelEncoder()

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df
        )
    
        self.model= lgbm.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.lr
        )

    def _build_prompt_forMCQ(self, df):
        #Combine prompt and options into a single text column
        df=df.copy()
        for opt in ["A","B","C","D","E"]:
          df["prompt"] += "[SEP]" +opt+": " + df[opt].fillna("")

        if "answer" in df.columns:
            return df[["prompt","answer"]]
        else:
            return df[["prompt"]]

    def fit(self, train_df):
        """
        Training
        """
        # Preprocess the DataFrame
        processed_df = self._build_prompt_forMCQ(train_df)
        texts = processed_df["prompt"]
        labels = processed_df["answer"]

        #encode labels
        y=self.LabelEncoder.fit_transform(labels)

        #vectorize texts
        X=self.vectorizer.fit_transform(texts)

        #train the model
        self.model.fit(X, y)

        return self

    def predict_top3(self, test_df):
        # Preprocess the DataFrame
        prompts_df = self._build_prompt_forMCQ(test_df)

        #vectorize texts
        X=self.vectorizer.transform(prompts_df['prompt'])

        #Get Probabilities
        probabilities = np.asanyarray(self.model.predict_proba(X))

        # converting to labels top3
        labels = self.LabelEncoder.classes_
        predictions = []
        for p in probabilities:
            top3_indices = np.argsort(p)[-3:][::-1]
            predictions.append(" ".join(labels[top3_indices]))

        return predictions


    def save(self, save_dir):
        """
        Helper to save the model, vectorizer and label encoder
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True) #sure dir exists

        joblib.dump(self.model, save_dir / f"{self.__class__.__name__}_model.pkl")
        joblib.dump(self.vectorizer, save_dir / f"{self.__class__.__name__}_vectorizer.pkl")
        joblib.dump(self.LabelEncoder, save_dir / f"{self.__class__.__name__}_label_encoder.pkl")
        with open(save_dir / f"{self.__class__.__name__}_config.json", "w") as f:
            json.dump(self.config, f, indent=4)

        print(f"Model, vectorizer and label encoder saved to {save_dir} successfully.")

    @classmethod
    def load(cls, model_dir):
        """
        Helper to load the saved model, vectorizer and label encoder to get instance.
        """
        model_dir = Path(model_dir)

        model = joblib.load(model_dir / f"{cls.__name__}_model.pkl")
        vectorizer = joblib.load(model_dir / f"{cls.__name__}_vectorizer.pkl")
        label_encoder = joblib.load(model_dir / f"{cls.__name__}_label_encoder.pkl")
        with open(model_dir / f"{cls.__name__}_config.json", "r") as f:
            config = json.load(f)

        instance = cls(**config)
        instance.model = model
        instance.vectorizer = vectorizer
        instance.LabelEncoder = label_encoder
        return instance

    @property
    def config(self):
        """
        Config to log hyperparameters
        """
        return {
            "lr": self.lr,
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df
        }