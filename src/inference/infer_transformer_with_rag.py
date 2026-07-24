import torch
import argparse
import pandas as pd
from pathlib import Path

from src.datamodules.rag_datamodule import RAGDataModule
from src.models.transformers.bert import BERT
from src.models.transformers.roberta import RoBERTa
from src.models.transformers.deberta_v3_base import DeBERTa_v3_base
from src.rag.knowledge_base import MCQKnowledgeBase
from src.rag.rag import RAGPipeline
from src.tokenizers.hf_tokenizer import HFTokenizer
from src.config import EnvConfig, RAGConfig
from scripts.make_processed_data_v001 import ensure_processed_data_exists_v001
from src.utils.kaggle_utils import load_kagglehub_model
from src.utils.prediction_utils import predict_top3

# Model configurations
MODELS={
    "BERT_RAG":{
        "model_class": BERT,
        "tokenizer_model_name": "bert-base-uncased",
        "model_handle": "dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver/pyTorch/bert",
        # "checkpoint_name": "BERT_RAG-epoch04-val_map30.9938.ckpt",
        "checkpoint_name": "BERT_RAG-epoch05-val_map30.8724.ckpt", #data v002

        # RAG specific configurations
        "rag_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "rag_reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 20,
        "rerank_k": 5

    },
    "RoBERTa_RAG":{
        "model_class": RoBERTa,
        "tokenizer_model_name": "roberta-base",
        "model_handle": "dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver/pyTorch/roberta",
        # "checkpoint_name": "RoBERTa_RAG-epoch03-val_map31.0000.ckpt",
        "checkpoint_name": "RoBERTa_RAG-epoch04-val_map30.8405.ckpt", #data v002

        # RAG specific configurations
        "rag_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "rag_reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 20,
        "rerank_k": 5
    },
    "DeBERTa_v3_base_RAG":{
        "model_class": DeBERTa_v3_base,
        "tokenizer_model_name": "microsoft/deberta-v3-base",
        "model_handle": "dhruvbansalup/dl-genai-project-26-t2-smart-mcq-solver/pyTorch/deberta_v3_base",
        # "checkpoint_name": "DeBERTa_v3_base_RAG-epoch02-val_map30.9773.ckpt",
        "checkpoint_name": "DeBERTa_v3_base_RAG-epoch02-val_map30.8624.ckpt",

        # RAG specific configurations
        "rag_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "rag_reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 20,
        "rerank_k": 5
    },
}

def infer_transformer_with_rag(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Loading Model
    model=load_kagglehub_model(model_handle=config["model_handle"], ckpt_name=config["checkpoint_name"], model_class=config["model_class"], load_to_device=device)

    # Get tokenized test data
    tokenizer = HFTokenizer(model_name=config["tokenizer_model_name"])

    ensure_processed_data_exists_v001()
    train_df=pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/train_v001.parquet")
    test_df=pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/test_v001.parquet")

    rag_pipeline=RAGPipeline(
        knowledge_base=MCQKnowledgeBase(),
        embedding_model=config["rag_embedding_model"],
        reranker_model=config["rag_reranker_model"],
        top_k=config["top_k"],
        rerank_k=config["rerank_k"]
    )

    datamodule = RAGDataModule(
        df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_length=RAGConfig.MAX_LENGTH,
        rag_pipeline=rag_pipeline,
        batch_size=8,
        num_workers=4,
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    print("=" * 60)
    print("INFERENCE RAG CONFIGURATION")
    print("=" * 60)
    print(f"Model          : {config['model_class'].__name__}")
    print(f"Embedding      : {config['rag_embedding_model']}")
    print(f"Reranker       : {config['rag_reranker_model']}")
    print(f"Top K          : {config['top_k']}")
    print(f"Rerank K       : {config['rerank_k']}")
    print("=" * 60)

    #Get predictions
    predictions=predict_top3(model, test_loader, device)
    print("Predictions Generated Successfully!")

    #Build submission
    submission_df = pd.DataFrame({
        "ID": test_df.index + 1,
        "Prediction": predictions
    })
    submission_df.to_csv(Path(EnvConfig.SUBMISSION_DIR) / "submission.csv", index=False)
    print(f"Submission file created at: {Path(EnvConfig.SUBMISSION_DIR) / 'submission.csv'}")

if __name__ == "__main__":

    #get cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    args = parser.parse_args()
    model_config = MODELS[args.model]

    infer_transformer_with_rag(model_config)