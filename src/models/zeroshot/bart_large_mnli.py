import torch
from transformers import pipeline

class BartLargeMNLI:
    """
    Zero-shot classification model using BART-large-MNLI from Hugging Face Transformers.
    """
    def __init__(self, device=None):

        if device is None:
            if torch.cuda.is_available():
                device = 0
            else:
                device = -1

        self.pipeline=pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device)

    def predict(self, prompt, candidate_labels):

        result = self.pipeline(
            sequences=prompt,
            candidate_labels=candidate_labels
        )

        label_map={
            candidate_labels[0]: "A",
            candidate_labels[1]: "B",
            candidate_labels[2]: "C",
            candidate_labels[3]: "D",
            candidate_labels[4]: "E"
        }
        answer=[]
        for ans in result['labels']:
            answer.append(label_map.get(ans))

        return " ".join(answer[:3])