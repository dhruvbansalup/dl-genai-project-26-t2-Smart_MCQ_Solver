import re
import torch

class WordTokenizer:
    """
        Tokenizes text to words and convert to indices using a given vocabulary.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize(self, text):
        #Convert text to tokens after removing punctuation and converting to lowercase
        text=str(text).lower()
        tokens = re.findall(r'\w+', text) # tokenize by words, remove punctuation
        return tokens

    def __encode(self, prompt, option, max_length):
        """
        Helper to encode single (prompt, option) pair to indices and attention mask.
        """

        # Build the text pair
        text = (
            f"[QUESTION]: {prompt} "
            f"[OPTION]: {option}"
        )

        #Convert text to indices
        tokens = self.tokenize(text)
        ids = [self.vocab.word_to_index(token) for token in tokens]

        ids=ids[:max_length] # Truncate to max_length
        ids += [0] * (max_length - len(ids)) #pad with 0s

        # attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1]*len(ids)
        pad_length = max_length - len(ids)
        ids += [0] * pad_length
        attention_mask += [0] * pad_length

        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }


    def encode_batch(self, prompt, options, max_length):
        """
        Encode batch of options for a given prompt.
        """
        # tokenized (prompt+option) pairs
        input_ids=[]
        # used to ignore the padding tokens
        attention_masks=[]

        for option in options:
            encoding=self.__encode(prompt, option, max_length)
            input_ids.append(encoding['input_ids'])
            attention_masks.append(encoding['attention_mask'])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long), # (5, L)
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long) # (5, L)
        }