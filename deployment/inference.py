import torch
from huggingface_hub import hf_hub_download
from model import BERT, HFTokenizer

LABELS=["A", "B", "C", "D", "E"]

class Solver:
    def __init__(self):
        # Download model weights from Hugging Face Hub
        weights_path = hf_hub_download(
            repo_id="dhruvbansalup/smart-mcq-solver-model",
            filename="bert_state_dict.pth",
        )

        # Model loading
        self.model = BERT()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)

        self.model.eval()

        # Tokenizer loading
        self.tokenizer = HFTokenizer(model_name="bert-base-uncased")

    @torch.inference_mode()
    def predict(self,prompt, options):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if next(self.model.parameters()).device != torch.device(device):
                    self.model.to(device)

        inputs = self.tokenizer.encode_batch(prompt, options)

        # unsqueeze to add batch dimension and move to device
        inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}

        logits = self.model(inputs)

        probs = torch.softmax(logits, dim=1)[0]

        top3 = torch.topk(probs, k=3)

        predictions ={
            LABELS[i]: float(prob)
            for i, prob in zip(
                top3.indices.tolist(),
                top3.values.tolist()
            )
        }

        return predictions

solver=Solver()