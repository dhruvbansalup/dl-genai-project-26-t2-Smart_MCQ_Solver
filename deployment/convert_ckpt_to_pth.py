# Utility to convert checkpoint to pth file for deployment

import torch

from src.models.transformers.bert import BERT

# Load Lightning checkpoint
model = BERT.load_from_checkpoint(
    "outputs/checkpoints/BERT-epoch02-val_map30.9950.ckpt",
    map_location="cpu"
)

# Save only the weights
torch.save(model.state_dict(), "bert_state_dict.pth")

# python -m deployment.convert_ckpt_to_pth