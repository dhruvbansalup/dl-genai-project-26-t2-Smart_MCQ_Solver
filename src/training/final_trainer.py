import argparse
import kagglehub
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from scripts.make_processed_data_v001 import ensure_processed_data_exists_v001
from src.config import EnvConfig, KaggleConfig, RAGConfig, GeneralConfig
from src.datamodules.rag_datamodule_02 import RAGDataModule02
from src.rag.rag import RAGPipeline
from src.rag.knowledge_base import MCQKnowledgeBase
from src.tokenizers.hf_tokenizer import HFTokenizer
from src.models.transformers.bert import BERT
from src.models.transformers.roberta import RoBERTa
from src.models.transformers.deberta_v3_base import DeBERTa_v3_base
from src.utils.time_utils import time_now_ist

def final_trainer(DATA_MODULE, MODEL,max_epochs, upload_kaggle=True, experiment_suffix=None):

    # setting for matrix multiplication for better performance
    torch.set_float32_matmul_precision('high')

    # Seeding for Reproducibility
    pl.seed_everything(GeneralConfig.SEED)

    model_class_name=MODEL.__class__.__name__
    if experiment_suffix:
        model_class_name += f"_{experiment_suffix}"

    #Model Checkpointing
    checkpoint_callbacks = [
        ModelCheckpoint(
            filename=f"Full-{model_class_name}-{{epoch:02d}}-{time_now_ist()}",
            dirpath=EnvConfig.CHECKPOINT_DIR,
            save_last=True,
            save_top_k=-1
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=None,
        num_sanity_val_steps=0, # skip sanity check to save time for large datasets
        limit_val_batches=0,
        callbacks=checkpoint_callbacks,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0, # clip gradients to avoid exploding gradients
        default_root_dir=EnvConfig.OUTPUT_DIR,
    )

    #Training
    print(f"Starting training for {model_class_name}...")
    trainer.fit(MODEL, DATA_MODULE)
    print(f"Training completed for {model_class_name}.")

    #Upload to Kagglehub
    if upload_kaggle:
        try:
            last_checkpoint_path = trainer.checkpoint_callback.last_model_path
            VARIATION = MODEL.__class__.__name__.lower()
            handle = f"{KaggleConfig.KAGGLE_USERNAME}/{KaggleConfig.KAGGLEHUB_MODEL_REPO}/pytorch/{VARIATION}"
            model_ref = kagglehub.model_upload(
                handle=handle,
                local_model_dir=last_checkpoint_path,
                version_notes=f"{MODEL.__class__.__name__} trained on {trainer.current_epoch} epochs at {time_now_ist()}",
            )

            print(f"{VARIATION} Model uploaded to Kagglehub Successfully!")
        except Exception as e:
            print(f"Error uploading model to Kagglehub: {e}")

    return trainer

def final_train_transformer_with_rag_full_data(model_class, tokenizer_model_name, epoch, embedding_model, reranker_model, top_k, rerank_k, uploadToKaggle=False, batch_size=4):
    ensure_processed_data_exists_v001()

    # Loading dataset
    train_df = pd.read_parquet(f"{EnvConfig.PROCESSED_DATA_DIR}/train_v001.parquet")

    # Initialize Tokenizer
    tokenizer = HFTokenizer(model_name=tokenizer_model_name)

    # RAG Pipeline
    knowledge_base = MCQKnowledgeBase()
    rag_pipeline = RAGPipeline(
        knowledge_base=knowledge_base,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        top_k=top_k,
        rerank_k=rerank_k
    )

    # Initialize Data Module
    data_module = RAGDataModule02(
        train_df=train_df,
        val_df=None,
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
    final_trainer(DATA_MODULE=data_module,
        MODEL=model,
        max_epochs=epoch,
        upload_kaggle=uploadToKaggle,
        experiment_suffix="RAG",
    )

if __name__ == "__main__":
    FINAL_MODELS={
        "BERT":{
            "model_class": BERT,
            "tokenizer_model_name": "bert-base-uncased",
            "epochs": 5,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 20,
            "rerank_k": 5,
            "batch_size": 4
        },
        "RoBERTa":{
            "model_class": RoBERTa,
            "tokenizer_model_name": "roberta-base",
            "epochs": 4,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 20,
            "rerank_k": 5,
            "batch_size": 4
        },
        "DeBERTa_v3_base":{
            "model_class": DeBERTa_v3_base,
            "tokenizer_model_name": "microsoft/deberta-v3-base",
            "epochs": 2,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 20,
            "rerank_k": 5,
            "batch_size": 2
        }
    }

    #get cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=FINAL_MODELS.keys(), required=True)
    parser.add_argument("--uploadToKaggle", action="store_true", help="Upload the trained model to Kagglehub")
    args = parser.parse_args()

    _model=FINAL_MODELS[args.model]

    final_train_transformer_with_rag_full_data(
        model_class=_model["model_class"],
        tokenizer_model_name=_model["tokenizer_model_name"],
        epoch=_model["epochs"],
        embedding_model=_model["embedding_model"],
        reranker_model=_model["reranker_model"],
        top_k=_model["top_k"],
        rerank_k=_model["rerank_k"],
        uploadToKaggle=args.uploadToKaggle,
        batch_size=_model["batch_size"])