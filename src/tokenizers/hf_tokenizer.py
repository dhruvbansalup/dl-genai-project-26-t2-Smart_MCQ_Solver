import torch
from transformers import AutoTokenizer

class HFTokenizer:
    """
    Hugging Face Models uses there own tokenizer, this class provides access to them.
    """
    def __init__(self, model_name: str):
        # Downloading pretrained tokenizer from HF
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_batch(self, prompt, options, max_length):

        encoding=self.tokenizer(
            [prompt]*len(options),
            options,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long), # (5, L)
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long) # (5, L)
        }