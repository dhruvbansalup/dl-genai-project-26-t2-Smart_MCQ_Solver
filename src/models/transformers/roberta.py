from src.models.transformers.hf_mcq_classifier import HFMCQClassifier

class RoBERTa(HFMCQClassifier):
    def __init__(
            self,
            lr=2e-5,
    ):
        super().__init__(lr=lr, model_name="roberta-base")