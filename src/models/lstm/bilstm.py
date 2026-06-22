import torch.nn as nn

from src.tokenizers.vocab import Vocabulary
from src.models.base_model import BaseModel
from src.models.layers.attention_pooling import AttentionPooling
from src.config import GeneralConfig

class BiLSTM01(BaseModel):

    """
    Architecture of BiLSTM01:

    No of Input=5 * (prompt+option)
    Input Text max_len=L
    Taken Batch Size= B
    dim of Embedding=E
    Hidden Size of LSTM=H

    Input Shape (5, B, L) -> Embedding i.e.convert to vectors (5*B, L, E) -> reshape to(batch,sq_len,input) (5*B, L, E) ->
    BiLSTM1 (5*B, L, H*2) -> BiLSTM2 (5*B, L, H*2) -> Pooling (compress seq_len to 1) (5*B, H*2) -> Classifier (B*5, 5) -> Reshape to (B, 5)
    """

    def __init__(
            self,
            lr,
            embedding_dim=128,
            hidden_size=256
        ):
        super().__init__(lr=lr)

        self.hidden_size = hidden_size
        self.num_options=GeneralConfig.NUM_OPTIONS #5

        #Embedding(tokens->dense vectors)
        self.embedding = nn.Embedding(
            num_embeddings=Vocabulary().__len__(),
            embedding_dim=embedding_dim,
            padding_idx=Vocabulary().word_to_index(Vocabulary().pad_token), #0
        )

        self.lstm=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True, #convert (seq_len, batch, input_size) to (batch, seq_len, input_size)
            dropout=0.3
        )

        #compress seq_len to 1
        self.pooling=AttentionPooling(hidden_size*2) 

        self.classifier=nn.Sequential(
            nn.Linear(hidden_size*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 5)
        )


        def forward(self, x):
            # (B, 5, L)
            batch_size, num_options, seq_len = x.shape

            # Reshape -> (B*5, L)
            x = x.view(batch_size * num_options, seq_len)

            # convert to vectors (B*5, L, E)
            x = self.embedding(x)

            # LSTM layers (B*5, L, 2H)
            x, _ = self.lstm(x)

            # Pooling (B*5, 2H, 1)
            x = self.pooling(x)

            # Final Classifier (B*5, 2H) -> (B*5, 5)
            scores = self.classifier(x)

            # Reshape to (B, 5)
            scores = scores.view(batch_size, num_options)

            return scores