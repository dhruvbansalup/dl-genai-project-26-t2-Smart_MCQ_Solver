import torch
from huggingface_hub import hf_hub_download

from src.models.transformers.bert import BERT
from src.tokenizers.hf_tokenizer import HFTokenizer

LABELS=["A", "B", "C", "D", "E"]

class Solver:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download model checkpoint from Hugging Face Hub
        checkpoint_path = hf_hub_download(
            repo_id="dhruvbansalup/smart-mcq-solver-model",
            filename="BERT-epoch02-val_map30.9950.ckpt",
        )

        # Model loading
        self.model = BERT.load_from_checkpoint(checkpoint_path, map_location=self.device)

        self.model.eval()
        self.model.to(self.device)

        # Tokenizer loading
        self.tokenizer = HFTokenizer(model_name="bert-base-uncased")

    @torch.inference_mode()
    def predict(self,prompt, options):
        inputs = self.tokenizer.encode_batch(prompt, options, max_length=242)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs)

        probs = torch.softmax(logits, dim=1)[0]

        top3 = torch.topk(probs, k=3)

        predictions ={
            LABELS[i]: float(prob)
            for i, prob in zip(
                top3.indices.tolist(), top3.values.tolist()
            )
        }

        return predictions

solver=Solver()