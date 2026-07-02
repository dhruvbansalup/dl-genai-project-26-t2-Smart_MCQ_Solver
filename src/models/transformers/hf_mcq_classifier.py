from transformers import AutoModel
from torch import nn

from src.models.base_model import BaseModel


class HFMCQClassifier(BaseModel):
    """
    Generic Hugging Face MCQ Classifier for transformer based models.
    """
    def __init__(
            self,
            model_name,
            lr
    ):
        super().__init__(lr=lr)

        self.model_name=model_name

        # Init the model from huggingface
        self.encoder=AutoModel.from_pretrained(self.model_name)

        # Set the encoder to train mode (by default, HF models are in eval mode)
        self.encoder.train()

        # getting transformer output embedding size for classifier input
        hidden_size=self.encoder.config.hidden_size

        self.classifier=nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):

        input_ids = x['input_ids']
        attention_mask = x['attention_mask']

        # (B, 5, L)
        batch_size, num_options, seq_len = input_ids.shape

        # Reshape -> (B*5, L)
        # Flatten option dim so every (prompt, option) pair is treated as a separate sequence
        input_ids = input_ids.reshape(batch_size * num_options, seq_len)
        attention_mask = attention_mask.reshape(batch_size * num_options, seq_len)

        # convert to token representations via transformer (B*5, L, E)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # get the pooled output (B*5, E)
        pooled=outputs.pooler_output

        # Final Classifier (B*5, E) -> (B*5, 1)
        logits=self.classifier(pooled)

        # Reshape to (B, 5)
        logits = logits.reshape(batch_size, num_options)

        return logits
