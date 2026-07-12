import torch
from torch.utils.data import Dataset

class MCQDataset(Dataset):
    """
    Returns a single data row from dataset
    tokenized (prompt+options) pairs
    """

    LABEL_MAP={"A":0, "B":1, "C":2, "D":3, "E":4}
    def __init__(self, df, tokenizer, max_length, return_labels=True):

        #init dataset & reset index to avoid index errors
        self.df=df.reset_index(drop=True)
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.return_labels=return_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row=self.df.iloc[idx]

        prompt=row['prompt']

        # In case of RAG, use context instead of prompt
        context = row.get("context", "")
        if context:
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question:\n{prompt}"
            )

        options=[row[o] for o in self.LABEL_MAP.keys()]

        x=self.tokenizer.encode_batch(prompt, options, max_length=self.max_length)

        if not self.return_labels:
            return x

        # for val/test
        y=self.LABEL_MAP[row['answer']]
        return x, y
