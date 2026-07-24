import torch
from huggingface_hub import hf_hub_download
from model import BERT, HFTokenizer

LABELS=["A", "B", "C", "D", "E"]

class Solver:
    def __init__(self):
        # Download model checkpoint from Hugging Face Hub
        checkpoint_path = hf_hub_download(
            repo_id="dhruvbansalup/smart-mcq-solver-model",
            filename="BERT-epoch02-val_map30.9950.ckpt",
        )

        # Model loading
        self.model = BERT.load_from_checkpoint(checkpoint_path)

        self.model.eval()

        # Tokenizer loading
        self.tokenizer = HFTokenizer(model_name="bert-base-uncased")

    @torch.inference_mode()
    def predict(self,prompt, options):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if next(self.model.parameters()).device != torch.device(device):
                    self.model.to(device)

        inputs = self.tokenizer.encode_batch(prompt, options, max_length=242)

        # unsqueeze to add batch dimension and move to device
        inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}

        logits = self.model(inputs)[0]

        top3_indices = torch.topk(logits, k=3).indices.tolist()

        return [LABELS[i] for i in top3_indices]

solver=Solver()