import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from abc import ABC, abstractmethod
from torchmetrics.classification import MulticlassAccuracy

import time

from src.config import GeneralConfig
from src.metrics.map3 import Map3

class BaseModel(pl.LightningModule, ABC):
    """
    Abstract Base class for all models
    """

    #TODO: Complete this

    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.num_classes = GeneralConfig.NUM_OPTIONS

        self.val_accuracy=MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy3=MulticlassAccuracy(num_classes=self.num_classes, top_k=3)

        # For tracking map@3
        self.map3_sum = 0.0
        self.map3_count = 0

        # For Epoch timings
        self.train_epoch_start_time = None
        self.val_epoch_start_time = None

    @abstractmethod
    def forward(self, x):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # Forward pass

        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_start(self):
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self):
        lr=self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=False, logger=True)

        # Logging duration
        if self.train_epoch_start_time is not None:
            epoch_duration = time.time() - self.train_epoch_start_time
            self.log("train_epoch_duration", epoch_duration, prog_bar=False, logger=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # Forward pass

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        preds=torch.argmax(y_hat, dim=1)

        self.val_accuracy.update(preds, y)
        self.val_accuracy3.update(y_hat, y)

        # For MAP@3
        batch_map3=Map3(y_hat, y)
        self.map3_sum += batch_map3*len(y)
        self.map3_count += len(y)

    def on_validation_epoch_start(self):
        self.map3_sum = 0.0
        self.map3_count = 0

        self.val_epoch_start_time = time.time()

    def on_validation_epoch_end(self):

        val_accuracy=self.val_accuracy.compute()
        self.log("val_accuracy", val_accuracy, prog_bar=True, logger=True)
        val_accuracy3=self.val_accuracy3.compute()
        self.log("val_accuracy3", val_accuracy3, prog_bar=True, logger=True)

        #MAP@3
        if self.map3_count > 0:
            val_map3 = self.map3_sum / self.map3_count
            self.log("val_map3", val_map3, prog_bar=True, logger=True)

        #Reset metrics
        self.val_accuracy.reset()
        self.val_accuracy3.reset()
        self.map3_sum = 0.0
        self.map3_count = 0

        # Logging duration
        if self.val_epoch_start_time is not None:
            epoch_duration = time.time() - self.val_epoch_start_time
            self.log("val_epoch_duration", epoch_duration, prog_bar=False, logger=True)