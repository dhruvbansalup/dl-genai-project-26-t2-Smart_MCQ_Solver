import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from src.config import GeneralConfig
from src.datasets.mcq_dataset import MCQDataset

class MCQDataModule02(pl.LightningDataModule):
    """
        Gives dataloaders for datasets
        Supporing saperate validation dataset, instead of splitting the training dataset into train and validation.
    """

    def __init__(
            self,
            train_df,
            val_df,
            test_df,
            tokenizer,
            max_length,
            batch_size=32,
            num_workers=4,
            fold=0
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df= val_df
        self.test_df = test_df

        self.tokenizer = tokenizer

        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Helper for GroupKFold
        self.fold=fold

    def setup(self, stage=None):
        self.train_dataset=None
        self.val_dataset=None
        self.test_dataset=None

        if self.train_df is not None:
            self.train_dataset = MCQDataset(self.train_df, self.tokenizer, self.max_length, return_labels=True)
        if self.val_df is not None:
            self.val_dataset = MCQDataset(self.val_df, self.tokenizer, self.max_length, return_labels=True)
        if self.test_df is not None:
            self.test_dataset = MCQDataset(self.test_df, self.tokenizer, self.max_length, return_labels=False)

    def train_dataloader(self):
        if self.train_df is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            #speedup
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_df is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.test_df is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )