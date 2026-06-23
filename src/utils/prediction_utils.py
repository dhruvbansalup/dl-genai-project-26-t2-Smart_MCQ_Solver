import torch
from tqdm import tqdm
def predict_top3(model, dataloader, device):
    LABELS = ["A", "B", "C", "D", "E"]
    predictions= []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            inputs = batch
            if isinstance(batch, (tuple, list)): # if contain (x,y) take only x
                inputs = batch[0]
            inputs = {key: value.to(device) for key, value in inputs.items()} # Set device
            logits = model(inputs) # predict
            top3_indices = torch.topk(logits, k=3, dim=1).indices #select top 3

            #convert to labels
            for row in top3_indices:
                predictions.append(" ".join(LABELS[index] for index in row.tolist()))

    return predictions