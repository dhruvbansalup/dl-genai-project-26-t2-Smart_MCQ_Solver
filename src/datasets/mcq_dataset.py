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

        option_tokens=[]
        attention_masks=[]
        for o in self.LABEL_MAP.keys():
            encoded=self.__build_pair(prompt, row[o])
            option_tokens.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        x={
            'input_ids': torch.tensor(option_tokens, dtype=torch.long), 'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }

        if not self.return_labels:
            return x

        # for val/test
        y=self.LABEL_MAP[row['answer']]
        return x, y

    def __build_pair(self, prompt, option):

        text = (
            f"[QUESTION]: {prompt} "
            f"[OPTION]: {option}"
        )

        return self.tokenizer.encode(text, max_length=self.max_length)