from collections import Counter

class Vocabulary:
    """
    Converts sentences to indices.
    2-way mapping b/w words & idx.
    """

    def __init__(self, min_freq=1, pad_token='<PAD>', unk_token='<UNK>'):

        self.min_freq = min_freq

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.special_tokens = [self.pad_token, self.unk_token]

        self.word2idx = {
            pad_token: 0,
            unk_token: 1
        }

        self.idx2word = {
            0: pad_token,
            1: unk_token
        }

        self.word_freq = Counter()

    def add_word(self, word):
        #Helper to add word
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word


    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                self.word_freq[word] += 1
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.add_word(word)

    def word_to_index(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_token])

    def index_to_word(self, idx):
        return self.idx2word.get(idx, '<UNK>')

    def __len__(self):
        return len(self.word2idx)