from transformers import AutoTokenizer

class HFTokenizer:
    """
    Hugging Face Models uses there own tokenizer, this class provides access to them.
    """
    def __init__(self, model_name: str):
        # Downloading pretrained tokenizer from HF
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, prompt, option, max_length):

        encoding=self.tokenizer(
            prompt,
            option,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }