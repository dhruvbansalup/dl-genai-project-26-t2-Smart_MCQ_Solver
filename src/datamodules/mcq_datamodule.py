import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import GeneralConfig
from src.datasets.mcq_dataset import MCQDataset

class MCQDataModule(pl.LightningDataModule):
    """
        Gives dataloaders for datasets
    """

    def __init__(
            self,
            df,
            test_df,
            tokenizer,
            max_length,
            batch_size=32,
            num_workers=4,
    ):
        super().__init__()

        self.df = df
        self.test_df = test_df

        self.tokenizer = tokenizer

        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        if self.df is not None:
            # Split df into train and val
            self.train_df, self.val_df = train_test_split(self.df, test_size=0.1, random_state=GeneralConfig.SEED, stratify=self.df['answer'])

            self.train_dataset = MCQDataset(self.train_df, self.tokenizer, self.max_length, return_labels=True)
            self.val_dataset = MCQDataset(self.val_df, self.tokenizer, self.max_length, return_labels=True)

        if self.test_df is not None:
            self.test_dataset = MCQDataset(self.test_df, self.tokenizer, self.max_length, return_labels=False)

    def train_dataloader(self):
        if self.df is None:
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
        if self.df is None:
            return None
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