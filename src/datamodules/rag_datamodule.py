from src.datamodules.mcq_datamodule import MCQDataModule
from src.datasets.mcq_dataset import MCQDataset
from src.rag.rag import RAGPipeline


class RAGDataModule(MCQDataModule):
    def __init__(
           self,
            df,
            test_df,
            tokenizer,
            max_length,
            rag_pipeline,
            batch_size=32,
            num_workers=4,
            fold=0,
    ):
        super().__init__(
            df=df,
            test_df=test_df,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            fold=fold
        )

        self.rag_pipeline = rag_pipeline

    def setup(self, stage=None):
        super().setup(stage)

        is_fit=stage in (None, "fit", "validate")
        is_test=stage in ("test", "predict")

        # Fit RAG Pipeline
        knowledge_df=None
        if is_fit:
            knowledge_df=self.train_df
        elif is_test:
            knowledge_df=self.df
        if knowledge_df is None:
            raise ValueError("No dataframe available for building the knowledge base.")
        self.rag_pipeline.fit(knowledge_df)


        # Get augmented datasets
        if (is_fit):
            if self.train_df is not None:
                self.train_df = self.rag_pipeline.augment_dataframe(self.train_df)
                self.val_df=self.rag_pipeline.augment_dataframe(self.val_df)

                self.train_dataset = MCQDataset(self.train_df, self.tokenizer, self.max_length, return_labels=True)
                self.val_dataset = MCQDataset(self.val_df, self.tokenizer, self.max_length, return_labels=True)
        elif (is_test):
            if self.test_df is not None:
                self.test_df=self.rag_pipeline.augment_dataframe(self.test_df)
                self.test_dataset = MCQDataset(self.test_df, self.tokenizer, self.max_length, return_labels=False)