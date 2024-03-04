from .constants import VOCAB_SIZE, VECTOR_SIZE, MAX_NORM, UNK_TOKEN
from .functions import requires_grad

import torch
import torch.nn as nn

from torch.nn.functional import cosine_similarity
from torchtext.vocab import Vocab

from typing import Tuple


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 vector_size: int = VECTOR_SIZE,
                 max_norm: float = MAX_NORM):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, vector_size, max_norm=max_norm)
        self.fc = nn.Sequential(
            nn.Linear(vector_size, vocab_size, bias=False),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.99 * self.fc(self.embedding(x)) + 1e-8


class Vector(nn.Module):
    def __init__(self, embedding: nn.Embedding, vocabulary: Vocab):
        assert embedding.num_embeddings == len(vocabulary)
        super().__init__()

        self.embedding = embedding
        self.vocabulary = vocabulary

        requires_grad(self, False)

    def nearest(self, vector: torch.Tensor) -> Tuple[str, torch.Tensor]:
        assert vector.size() == self.forward(UNK_TOKEN).size()
        vector = vector.unsqueeze(0)

        similarities = {
            word: cosine_similarity(self.forward(word).unsqueeze(0), vector)
            for word in self.vocabulary.get_itos()
        }

        nearest_word = max(similarities, key=similarities.get)
        return nearest_word, similarities[nearest_word]

    def forward(self, word: str) -> torch.Tensor:
        index = torch.tensor(self.vocabulary[word])
        return self.embedding(index)
