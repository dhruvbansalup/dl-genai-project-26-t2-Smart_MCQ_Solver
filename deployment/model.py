# COPY OF MODEL & TOKENIZER FOR DEPLOYMENT

from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch

class BERT(nn.Module):
    def __init__(self):
        super().__init__()

        # Init the model from huggingface
        self.encoder=AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float32)

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
        pooled=outputs.last_hidden_state[:, 0, :]

        # Final Classifier (B*5, E) -> (B*5, 1)
        logits=self.classifier(pooled)

        # Reshape to (B, 5)
        logits = logits.reshape(batch_size, num_options)

        return logits


class HFTokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_batch(self, prompt, options):

        encoding=self.tokenizer(
            [prompt]*len(options),
            options,
            truncation=True,
            padding='max_length',
            max_length=242,
            return_attention_mask=True
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long), # (5, L)
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long) # (5, L)
        }