import torch
import pandas as pd
from pathlib import Path

from scripts.make_processed_data_v001 import ensure_processed_data_exists
from src.config import EnvConfig, GeneralConfig
from src.tokenizers.hf_tokenizer import HFTokenizer
from src.utils.kaggle_utils import load_kagglehub_model
from src.inference.infer_transformer import MODELS as models_config
from src.utils.prediction_utils import predict_logits, predict_top3_from_logits
from src.datamodules.mcq_datamodule import MCQDataModule

def get_logits_from_transformer_model(model_handle, checkpoint_name, model_class, tokenizer_model_name, test_df, device):
    """
    Helper to get logits from a specific model
    """

    # Loading Model
    model=load_kagglehub_model(model_handle=model_handle, ckpt_name=checkpoint_name, model_class=model_class, load_to_device=device)

    # Get tokenized test data
    tokenizer = HFTokenizer(model_name=tokenizer_model_name)

    datamodule = MCQDataModule(
        df=None,
        test_df=test_df,
        tokenizer=tokenizer,
        max_length=GeneralConfig.MAX_LENGTH*2+2,# we have pairs, and 2 extra tokens for [QUESTION] and [OPTION],
        batch_size=8,
        num_workers=4,
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    #Get prediction logits
    logits = predict_logits(model, test_loader, device)
    print(f"Logits generated successfully for model: {model_handle}")
    return logits

def infer_ensemble():
    '''
    Combining predictions of BERT+RoBERTa+DeBERTa_v3_base models to generate ensemble predictions.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Loading Data
    ensure_processed_data_exists()
    test_df=pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet")

    #Getting prediction logits from each model
    bert_logits=get_logits_from_transformer_model(
        model_handle=models_config["BERT"]["model_handle"],
        checkpoint_name=models_config["BERT"]["checkpoint_name"],
        model_class=models_config["BERT"]["model_class"],
        tokenizer_model_name=models_config["BERT"]["tokenizer_model_name"],
        test_df=test_df, device=device)
    roberta_logits = get_logits_from_transformer_model(
        model_handle=models_config["RoBERTa"]["model_handle"],
        checkpoint_name=models_config["RoBERTa"]["checkpoint_name"],
        model_class=models_config["RoBERTa"]["model_class"],
        tokenizer_model_name=models_config["RoBERTa"]["tokenizer_model_name"],
        test_df=test_df, device=device)
    deberta_logits = get_logits_from_transformer_model(
        model_handle=models_config["DeBERTa_v3_base"]["model_handle"],
        checkpoint_name=models_config["DeBERTa_v3_base"]["checkpoint_name"],
        model_class=models_config["DeBERTa_v3_base"]["model_class"],
        tokenizer_model_name=models_config["DeBERTa_v3_base"]["tokenizer_model_name"],
        test_df=test_df, device=device)

    #Converting to probabilities
    bert_probs = torch.softmax(bert_logits, dim=1)
    roberta_probs = torch.softmax(roberta_logits, dim=1)
    deberta_probs = torch.softmax(deberta_logits, dim=1)

    ensemble_probabilities=(
        0.4*bert_probs +
        0.4*deberta_probs +
        0.2*roberta_probs
    )

    #Getting top3 predictions from ensemble logits
    ensemble_predictions=predict_top3_from_logits(ensemble_probabilities)

    #Build submission
    #Build submission
    submission_df = pd.DataFrame({
        "ID": test_df.index + 1,
        "Prediction": ensemble_predictions
    })
    submission_df.to_csv(Path(EnvConfig.SUBMISSION_DIR) / "submission.csv", index=False)
    print(f"Submission file created at: {Path(EnvConfig.SUBMISSION_DIR) / 'submission.csv'}")

if __name__ == "__main__":
    infer_ensemble()