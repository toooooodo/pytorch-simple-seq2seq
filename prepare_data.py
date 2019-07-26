from __future__ import unicode_literals, print_function, division
from torch.utils.data import Dataset, DataLoader
from io import open
import unicodedata
import re
import random
import numpy as np


class Lang:
    def __init__(self, name):
        self.name = name
        # {index: token}
        self.index_to_token = {0: '<pad>', 1: '<bos>', 2: '<eos>'}
        # {token: index}
        self.token_to_index = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
        # {token: count_of_this_token}
        self.token_count = dict()
        self.token_n = 3  # number of different tokens

    def add_sentence(self, sentence):
        for token in sentence.split(' '):
            self.add_token(token)

    def add_token(self, token):
        if token in self.token_to_index:
            self.token_count[token] += 1
        else:
            self.index_to_token[self.token_n] = token
            self.token_to_index[token] = self.token_n
            self.token_n += 1
            self.token_count[token] = 1


class F2EDataSet(Dataset):
    def __init__(self, max_length=10):
        super(F2EDataSet, self).__init__()
        self.max_length = max_length
        self.eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        self.in_lang, self.out_lang, self.in_seq, self.out_seq = self.load_text()

    def __len__(self):
        return self.in_seq.shape[0]

    def __getitem__(self, item):
        """
        :param item:
        :return: [French Sentence, English Sentence]
        """
        return [self.in_seq[item], self.out_seq[item]]

    def load_text(self):
        with open('./data/eng-fra.txt', 'r', encoding='utf-8') as f:
            pairs = f.readlines()
        # pair[0]: ['go .', 'va !'] English => French
        pairs = [[self.normalizeString(s) for s in pair.rstrip().split('\t')] for pair in pairs]
        # French => English
        pairs = [list(reversed(pair)) for pair in pairs]
        print(f'Read {len(pairs)} sentence pairs.')
        pairs = [pair for pair in pairs if self.filter_pair(pair)]
        print(f'Trimmed to {len(pairs)} sentence pairs.')
        in_language = Lang('French')
        out_language = Lang('English')
        for in_sentence, out_sentence in pairs:
            in_language.add_sentence(in_sentence)
            out_language.add_sentence(out_sentence)
        print(in_language.name, in_language.token_n)
        print(out_language.name, out_language.token_n)
        in_indices, out_indices = [], []
        for in_sentence, out_sentence in pairs:
            in_indices.append(self.convert_token_to_index(in_language, in_sentence))
            out_indices.append(self.convert_token_to_index(out_language, out_sentence))
        in_indices, out_indices = np.array(in_indices), np.array(out_indices)
        return in_language, out_language, in_indices, out_indices

    def unicodeToAscii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def filter_pair(self, pair):
        return len(pair[0].split(' ')) < self.max_length and len(pair[1].split(' ')) < self.max_length and pair[
            1].startswith(self.eng_prefixes)

    def convert_token_to_index(self, lang, sentence):
        indices = []
        for token in sentence.split(' '):
            indices.append(lang.token_to_index[token])
        # padding
        indices += [2] + [0] * (self.max_length - len(indices) - 1)
        return indices


if __name__ == '__main__':
    data_set = F2EDataSet()
    loader = DataLoader(data_set, batch_size=32, shuffle=True)
    for batch_idx, (in_seq, out_seq) in enumerate(loader):
        print(in_seq[0].dtype)
        print(out_seq[0])
        break
