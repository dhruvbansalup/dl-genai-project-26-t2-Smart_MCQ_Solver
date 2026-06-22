import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """
        Get weighted sum of outputs using attention scores.
        Used to compress sequence of hidden states into a single vector representation.
    """
    def __init__(self, input_dim):
        super().__init__()

        # Weights (W)
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):

        scores = self.attention_weights(x)
        attention_weights = torch.softmax(scores, dim=1)

        # A=Softmax(W*x)
        pooled_output = torch.sum(attention_weights * x, dim=1)

        # A*X (dim=1)
        return pooled_output
