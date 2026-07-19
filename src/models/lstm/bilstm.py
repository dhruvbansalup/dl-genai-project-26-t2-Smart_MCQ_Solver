import torch.nn as nn

from src.models.base_model import BaseModel
from src.models.lstm.attention_pooling import AttentionPooling
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
    BiLSTM1 (5*B, L, H*2) -> BiLSTM2 (5*B, L, H*2) -> Pooling (compress seq_len to 1) (5*B, H*2) -> Classifier (B*5, 1) -> Reshape to (B, 5)
    """

    def __init__(
            self,
            lr,
            vocab_size,
            embedding_dim=128,
            hidden_size=256
        ):
        super().__init__(lr=lr)

        self.hidden_size = hidden_size
        self.num_options=GeneralConfig.NUM_OPTIONS #5

        #Embedding(tokens->dense vectors)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0, # <PAD> maps to 0 in vocab
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

            nn.Linear(64, 1)
        )


    def forward(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']

        # (B, 5, L)
        batch_size, num_options, seq_len = input_ids.shape

        # Reshape -> (B*5, L)
        x = input_ids.reshape(batch_size * num_options, seq_len)
        attention_mask = attention_mask.reshape(batch_size * num_options, seq_len)

        # convert to vectors (B*5, L, E)
        x = self.embedding(x)

        # LSTM layers (B*5, L, 2H)
        x, _ = self.lstm(x)

        # Pooling (B*5, 2H, 1)
        x = self.pooling(x, attention_mask=attention_mask)

        # Final Classifier (B*5, 2H) -> (B*5, 1)
        scores = self.classifier(x)

        # Reshape to (B, 5)
        scores = scores.reshape(batch_size, num_options)

        return scores
    


    # Vocab persistance helpers for inference

    def attach_vocab(self, vocab):
        #Helper to attach vocab to model for encoding during inference
        self.word2idx=vocab.word2idx

    def on_save_checkpoint(self, checkpoint):
        #Helper to save vocab along with model checkpoint
        checkpoint['word2idx'] = self.word2idx

    def on_load_checkpoint(self, checkpoint):
        #Helper to load vocab along with model checkpoint
        self.word2idx = checkpoint['word2idx']

    @property
    def vocab(self):
        #Helper to get vocab from model via model.vocab
        from src.tokenizers.vocab import Vocabulary
        return Vocabulary.from_word2idx(self.word2idx)