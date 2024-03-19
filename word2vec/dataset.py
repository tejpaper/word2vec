import itertools
import json
from collections import Counter, OrderedDict
from string import whitespace, digits, punctuation

import torch
from torch.utils.data import TensorDataset
from torchtext.data import get_tokenizer, custom_replace
from torchtext.vocab import vocab

from tqdm.autonotebook import tqdm


def load_amazon_fashion(path: str, verbose: bool = True) -> list[str]:
    with open(path) as file:
        samples = [review.get('reviewText', '') for review in
                   tqdm(list(map(json.loads, file)), disable=not verbose, desc='Loading dataset')]
    samples = list(itertools.chain(*map(lambda x: x.split('.'), samples)))
    samples = [sample for sample in samples if sample]
    return samples


class Dataset(TensorDataset):
    def __init__(self,
                 samples: list[str],
                 skip_gram_size: int,
                 negative_samples_num: int,
                 vocab_size: int,
                 unk_token: str,
                 is_lower: bool,
                 are_digits: bool,
                 is_punctuation: bool,
                 verbose: bool = True,
                 ) -> None:

        self.samples = samples
        self.skip_grams = set(range(-skip_gram_size // 2 + 1, skip_gram_size // 2 + 1)) - {0}
        self.negative_samples_num = negative_samples_num
        self.unk_token = unk_token

        self.unk_index = None
        self.frequencies = None
        self.vocabulary = None
        self._prepare_seq_dataset(vocab_size, is_lower, are_digits, is_punctuation, verbose)
        input_indexes, target_indexes = self._create_blocks(verbose)

        super().__init__(input_indexes, target_indexes)

    def _prepare_seq_dataset(self,
                             vocab_size: int,
                             is_lower: bool,
                             are_digits: bool,
                             is_punctuation: bool,
                             verbose: bool,
                             ) -> None:
        replace_patterns = {(char, ' ') for char in whitespace}

        if not are_digits:
            replace_patterns |= {(char, '') for char in digits}
        if not is_punctuation:
            replace_patterns |= {(fr'\{char}', '') for char in punctuation}

        transforms = custom_replace(replace_patterns)
        tokenizer = get_tokenizer((None, 'basic_english')[is_lower])

        self.samples = [tokenizer(sample) for sample in tqdm(
            transforms(self.samples),
            total=len(self.samples),
            disable=not verbose,
            desc='Preparing dataset')]
        self.samples = [sample for sample in self.samples if len(sample) > 1]

        counter = Counter(itertools.chain(*self.samples))

        tokens_total = sum(map(lambda x: x[1], counter.most_common()))
        counter = counter.most_common(vocab_size - 1)

        in_vocab_total = sum(map(lambda x: x[1], counter))
        unk_total = tokens_total - in_vocab_total

        for i, (_, freq) in enumerate(counter):
            if unk_total > freq:
                self.unk_index = i
                break
        else:
            self.unk_index = vocab_size - 1

        counter.insert(self.unk_index, (self.unk_token, unk_total))

        self.frequencies = torch.tensor([freq for _, freq in counter])
        self.frequencies = self.frequencies ** (3 / 4)
        self.frequencies /= self.frequencies.sum()

        self.vocabulary = vocab(OrderedDict(counter))
        self.vocabulary.set_default_index(self.unk_index)

    def _create_blocks(self, verbose: bool) -> tuple[torch.Tensor, torch.Tensor]:
        input_indexes = list()
        target_indexes = list()

        for sample in tqdm(self.samples, disable=not verbose, desc='Creating blocks'):
            encoded = [self.vocabulary[token] for token in sample]
            tokens_num = len(encoded)

            for i, x_index in enumerate(encoded):
                if x_index == self.unk_index:
                    continue

                for shift in self.skip_grams:
                    j = i + shift
                    if 0 <= j < tokens_num and encoded[j] != self.unk_index:
                        input_indexes.append(x_index)
                        target_indexes.append(encoded[j])

        return torch.tensor(input_indexes), torch.tensor(target_indexes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_idx, target_idx = super().__getitem__(index)
        output_indexes = self.frequencies.multinomial(self.negative_samples_num + 1)
        output_indexes[0] = target_idx
        return input_idx, output_indexes
