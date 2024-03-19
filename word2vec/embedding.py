from word2vec.constants import UNK_TOKEN

import typing

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torchtext.vocab import Vocab


class Word2Vec(nn.Embedding):
    def __init__(self,
                 vocab_size: int,
                 vector_size: int,
                 max_norm: float,
                 vocabulary: Vocab,
                 unk_token: str = UNK_TOKEN,
                 ) -> None:
        super().__init__(vocab_size, vector_size, max_norm=max_norm)
        self.vocabulary = vocabulary
        self.unk_token = unk_token

    def forward(self, x: str | typing.Sequence[str] | torch.Tensor) -> torch.Tensor:
        if isinstance(x, str):
            x = [x]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(list(map(self.vocabulary.__getitem__, x)), device=self.weight.device)
        return super().forward(x)

    def nearest(self, vector: torch.Tensor, k: int = 1) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        scores, indexes = torch.topk(cosine_similarity(self.weight, vector), k)
        words = list(map(self.vocabulary.get_itos().__getitem__, indexes))
        vectors = self.weight[indexes]
        return words, vectors, scores


class SkipGramModel(nn.Module):
    def __init__(self, word2vec: Word2Vec) -> None:
        super().__init__()
        self.word2vec = word2vec
        self.fc = nn.Linear(*word2vec.weight.shape[::-1], bias=False)

    def forward(self, x: torch.Tensor, indexes: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.word2vec(x)

        if indexes is None:
            return self.fc(embeddings)
        else:
            return torch.matmul(self.fc.weight[indexes], embeddings.unsqueeze(-1)).squeeze(-1)
