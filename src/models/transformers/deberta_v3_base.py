import torch
from src.models.transformers.hf_mcq_classifier import HFMCQClassifier

class DeBERTa_v3_base(HFMCQClassifier):
    def __init__(
            self,
            lr=2e-5,
            weight_decay=0.01,
    ):
        super().__init__(lr=lr, model_name="microsoft/deberta-v3-base")

        self.weight_decay = weight_decay

    def configure_optimizers(self):
        # Overriding the optimizer to include weight decay (to keep weights small and prevent overfitting)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer