from .constants import DEVICE, SKIP_GRAM_SIZE, NEGATIVE_S_NUM, VOCAB_SIZE, UNK_TOKEN

import json
import random
import itertools

from collections import Counter, OrderedDict
from string import whitespace, digits, punctuation

import torch

from torch.utils.data import Dataset
from torchtext.data import get_tokenizer, custom_replace
from torchtext.vocab import vocab

from typing import Generator, Tuple


class DataSet(Dataset):
    def __init__(self, json_file: str,
                 initial_seed: int = None,
                 device: torch.device = DEVICE,
                 skip_gram_size: int = SKIP_GRAM_SIZE,
                 negative_samples_num: int = NEGATIVE_S_NUM,
                 vocab_size: int = VOCAB_SIZE,
                 unk_token: str = UNK_TOKEN,
                 is_lower: bool = True,
                 is_digits: bool = True,
                 is_punctuation: bool = False):

        self.device = device
        self.skip_grams = set(range(-skip_gram_size // 2 + 1, skip_gram_size // 2 + 1)) - {0}
        self.negative_samples_num = negative_samples_num

        replace_patterns = {(char, ' ') for char in whitespace}
        if not is_digits:
            replace_patterns |= {(char, '') for char in digits}
        if not is_punctuation:
            replace_patterns |= {(fr'\{char}', '') for char in punctuation}

        transforms = custom_replace(replace_patterns)
        tokenizer = get_tokenizer((None, 'basic_english')[is_lower])

        self.quotes = set()

        with open(json_file) as file:
            for review in map(json.loads, file):
                sentences = review.get('reviewText', '').split('.')
                sentences = transforms(sentences)

                self.quotes |= set((*tokenizer(sentence),) for sentence in sentences)
        self.quotes = [sentence for sentence in self.quotes if len(sentence) > 1]
        self.shuffle(initial_seed)

        counter = Counter(itertools.chain(*self.quotes))
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        tokens_num = sum([freq for _, freq in counter])
        counter = counter[:vocab_size - 1]

        in_vocab_tokens_num = sum([freq for _, freq in counter])
        unk_freq = tokens_num - in_vocab_tokens_num

        for i, (_, freq) in enumerate(counter):
            if unk_freq > freq:
                self.unk_index = i
                break
        else:
            self.unk_index = vocab_size - 1

        counter.insert(self.unk_index, (unk_token, unk_freq))

        self.frequencies = torch.tensor([freq for _, freq in counter])
        self.frequencies = self.frequencies ** (3 / 4)
        self.frequencies /= self.frequencies.sum()

        self.vocabulary = vocab(OrderedDict(counter))
        self.vocabulary.set_default_index(self.unk_index)

    def shuffle(self, seed: int = None):
        self.quotes.sort()
        random.seed(seed)
        random.shuffle(self.quotes)

    def __len__(self) -> int:
        return len(self.quotes)

    def __getitem__(self, index: int) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        sentence = [torch.tensor(self.vocabulary[token], device=self.device) for token in self.quotes[index]]

        for i, x_index in enumerate(sentence):
            if x_index == self.unk_index:
                continue

            targets_indices = list()

            for j, t_index in enumerate(sentence):
                if i - j in self.skip_grams and t_index != self.unk_index:
                    targets_indices.append(t_index)

            targets_num = len(targets_indices)

            if not targets_num:
                continue

            targets_indices = torch.tensor(targets_indices, device=self.device)
            negative_indices = self.frequencies.multinomial(self.negative_samples_num * targets_num).to(self.device)

            yield x_index, targets_indices, negative_indices
