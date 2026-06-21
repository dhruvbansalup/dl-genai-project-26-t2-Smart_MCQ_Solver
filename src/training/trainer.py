import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.config import GeneralConfig, EnvConfig, WandbConfig
from src.utils.time_utils import time_now_ist

def train(DATA_MODULE, MODEL,max_epochs, log=True, upload_kaggle=True):

    # setting for matrix multiplication for better performance
    torch.set_float32_matmul_precision('high')

    # Seeding for Reproducibility
    pl.seed_everything(GeneralConfig.SEED)

    model_class_name=MODEL.__class__.__name__

    #Model Checkpointing
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val_map3",
            mode="max",
            filename=f"{model_class_name}-{{epoch:02d}}-{{val_map3:.4f}}",
            dirpath=EnvConfig.CHECKPOINT_DIR,
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_map3",
            mode="max",
            patience=5,
        )
    ]

    wandb_logger=None
    if log:
        from src.utils.wandb_utils import login_wandb
        login_wandb()
        wandb_logger = WandbLogger(
            project=WandbConfig.PROJECT_NAME,
            name=f"{model_class_name}-{time_now_ist()}",
            save_dir=EnvConfig.OUTPUT_DIR,
            log_model=True
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callbacks,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=10,
        enable_progress_bar=True,
        accelerator="auto",
        devices="auto",
    )

    #Training
    print(f"Starting training for {model_class_name}...")
    trainer.fit(MODEL, DATA_MODULE)
    print(f"Training completed for {model_class_name}.")

    #Upload to Kagglehub
    if upload_kaggle:
        from src.utils.kaggle_utils import upload_to_kagglehub
        upload_to_kagglehub(MODEL,trainer)

    #Close Wandb Logger
    if wandb_logger:
        wandb_logger.experiment.finish()

    return trainer