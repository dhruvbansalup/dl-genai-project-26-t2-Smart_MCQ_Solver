import re

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

    def encode(self, text, max_length):
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