import argparse
import pandas as pd

from scripts.make_processed_data_v002 import ensure_processed_data_exists_v002
from src.config import EnvConfig, RAGConfig
from src.datamodules.rag_datamodule import RAGDataModule
from src.datamodules.rag_datamodule_02 import RAGDataModule02
from src.rag.rag import RAGPipeline
from src.rag.knowledge_base import MCQKnowledgeBase
from src.tokenizers.hf_tokenizer import HFTokenizer
from src.training.trainer import train
from src.models.transformers.bert import BERT
from src.models.transformers.roberta import RoBERTa
from src.models.transformers.deberta_v3_base import DeBERTa_v3_base

def train_transformer_with_rag_data_v002(model_class, tokenizer_model_name, log=False, uploadToKaggle=False, batch_size=4):
    ensure_processed_data_exists_v002()

    # Loading dataset
    train_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/train_v002.parquet")
    val_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/val_v002.parquet")

    # Initialize Tokenizer
    tokenizer = HFTokenizer(model_name=tokenizer_model_name)

    # RAG Pipeline
    knowledge_base = MCQKnowledgeBase()
    rag_pipeline = RAGPipeline(
        knowledge_base=knowledge_base,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=20,
        rerank_k=5
    )

    # Initialize Data Module
    data_module = RAGDataModule02(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        tokenizer=tokenizer,
        max_length=RAGConfig.MAX_LENGTH,
        rag_pipeline=rag_pipeline,
        batch_size=batch_size,
        num_workers=4,
    )

    # Initialize Model
    model = model_class(lr=2e-5)

    # Print RAG Configuration
    print("=" * 60)
    print("RAG CONFIGURATION")
    print("=" * 60)
    print(f"Knowledge Base : {knowledge_base.__class__.__name__}")
    print(f"Embedding      : {rag_pipeline.embedding_model}")
    print(f"Reranker       : {rag_pipeline.reranker_model}")
    print(f"Top K          : {rag_pipeline.top_k}")
    print(f"Rerank K       : {rag_pipeline.rerank_k}")
    print(f"Max Length     : {RAGConfig.MAX_LENGTH}")
    print("=" * 60)

    # Train the model
    train(DATA_MODULE=data_module,
        MODEL=model,
        max_epochs=15,
        log=log,
        upload_kaggle=uploadToKaggle,
        experiment_suffix="RAG",
        config_logs={
            "pipeline": "RAG",
            "knowledge_base": knowledge_base.__class__.__name__,
            "embedding_model": rag_pipeline.embedding_model,
            "reranker_model": rag_pipeline.reranker_model,
            "retrieval_top_k": rag_pipeline.top_k,
            "retrieval_rerank_k": rag_pipeline.rerank_k,
            "max_length": RAGConfig.MAX_LENGTH,
            "data_version": "v002"
        }
    )

if __name__ == "__main__":

    MODELS={
        # model_name: (model_class, tokenizer_model_name)
        "BERT":(BERT, "bert-base-uncased"),
        "RoBERTa":(RoBERTa, "roberta-base"),
        "DeBERTa_v3_base":(DeBERTa_v3_base, "microsoft/deberta-v3-base"),
    }

    #get cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=MODELS.keys(), required=True)
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--uploadToKaggle", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    model_class, tokenizer_model_name=MODELS[args.model]

    train_transformer_with_rag_data_v002(model_class=model_class, tokenizer_model_name=tokenizer_model_name, log=args.log, uploadToKaggle=args.uploadToKaggle, batch_size=args.batch_size)