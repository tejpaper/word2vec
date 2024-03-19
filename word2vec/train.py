from word2vec.constants import *
from word2vec.dataset import load_amazon_fashion, Dataset
from word2vec.embedding import Word2Vec, SkipGramModel

import os
import time
import typing
from dataclasses import dataclass

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm


@dataclass
class TrainingConfig:
    # files
    path2data: str = PATH2DATA
    checkpoints_dir: str = CHECKPOINTS_DIR
    logs_file: str = LOGS_FILE
    test_file: str = TEST_FILE

    # dataset
    skip_gram_size: int = 5
    negative_s_num: int = 5
    unk_token: str = UNK_TOKEN
    is_lower: bool = True
    are_digits: bool = True
    is_punctuation: bool = False

    # model
    vocab_size: int = 4000
    vector_size: int = 300
    max_norm = 1

    # training
    batch_size: int = 64
    lr: float = 0.025
    final_lr: float = 1e-8
    step_size: int = 1000
    epochs: int = 5
    log_freq: int = 450  # in seconds

    # global
    device: torch.device = DEVICE
    verbose: bool = True


def training_loop(model: SkipGramModel,
                  config: TrainingConfig,
                  dl: DataLoader,
                  optimizer: Adam,
                  scheduler: StepLR,
                  initial_epoch: int = 1,
                  ) -> typing.Iterator[tuple[torch.Tensor, int]]:
    assert 1 <= initial_epoch <= config.epochs

    starting_point = time.time()
    losses = list()

    for epoch in range(initial_epoch, config.epochs + 1):
        for input_indexes, output_indexes in tqdm(dl, desc='Batches', disable=not config.verbose):
            input_indexes = input_indexes.to(config.device)
            output_indexes = output_indexes.to(config.device)

            optimizer.zero_grad()

            preds = model(input_indexes, output_indexes)
            targets = torch.zeros_like(input_indexes, device=config.device)
            loss = cross_entropy(preds, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if time.time() - starting_point >= config.log_freq:
                yield sum(losses) / len(losses), epoch
                starting_point = time.time()
                losses = list()


@torch.no_grad()
def test(word2vec: Word2Vec, test_samples: list[str], verbose: bool = False) -> tuple[float, float]:
    training = word2vec.training
    word2vec.eval()

    top5_accuracy = 0
    similarity = 0

    for minuend, subtrahend, term, target in test_samples:
        vector = word2vec(minuend) - word2vec(subtrahend) + word2vec(term)
        neighbors, _, scores = word2vec.nearest(vector, k=5)

        if verbose:
            print(f'{minuend} - {subtrahend} + {term} = {target} | {neighbors[0]}')

        if target in neighbors:
            top5_accuracy += 1
            similarity += scores[neighbors.index(target)]
        else:
            similarity += scores[0].item()

    top5_accuracy /= len(test_samples)
    similarity /= len(test_samples)

    word2vec.training = training
    return top5_accuracy, similarity


def get_checkpoint(config: TrainingConfig) -> tuple[SkipGramModel, DataLoader, list[str], Adam, StepLR, int, int]:
    indexes = [int(f[11:-4]) for f in os.listdir(config.checkpoints_dir) if f.endswith('.pth')] + [-1, 0]
    checkpoint_num = max(indexes)

    ds_kwargs = dict(
        samples=load_amazon_fashion(config.path2data, config.verbose),
        skip_gram_size=config.skip_gram_size,
        negative_samples_num=config.negative_s_num,
        vocab_size=config.vocab_size,
        unk_token=config.unk_token,
        is_lower=config.is_lower,
        are_digits=config.are_digits,
        is_punctuation=config.is_punctuation,
        verbose=config.verbose,
    )

    if not checkpoint_num:
        initial_epoch = 1

        ds = Dataset(**ds_kwargs)
        word2vec = Word2Vec(config.vocab_size,
                            config.vector_size,
                            config.max_norm,
                            ds.vocabulary,
                            config.unk_token)
        model = SkipGramModel(word2vec).to(config.device)

        optimizer = Adam(model.parameters(), config.lr)
        scheduler = StepLR(
            optimizer, config.step_size,
            gamma=(config.final_lr / config.lr) ** (config.step_size / (config.epochs * len(ds) / config.batch_size))
        )
    else:
        filename = f'checkpoint-{checkpoint_num}.pth'
        checkpoint_dict = torch.load(os.path.join(config.checkpoints_dir, filename), map_location=config.device)

        initial_epoch = checkpoint_dict['initial_epoch']

        ds = Dataset(**ds_kwargs)
        model = checkpoint_dict['model']

        optimizer = checkpoint_dict['optimizer']
        scheduler = checkpoint_dict['scheduler']

    dl = DataLoader(
        dataset=ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=next(model.parameters()).is_cuda,
    )

    with open(config.test_file) as file:
        test_samples = [
            line for line in map(lambda t: t.split(), file.readlines())
            if all(map(lambda word: word in model.word2vec.vocabulary, line))
        ]

    return model, dl, test_samples, optimizer, scheduler, checkpoint_num, initial_epoch


def train(model: SkipGramModel,
          config: TrainingConfig,
          dl: DataLoader,
          test_samples: list[str],
          optimizer: Adam,
          scheduler: StepLR,
          checkpoint_num: int,
          initial_epoch: int) -> None:

    training_iter = training_loop(model, config, dl, optimizer, scheduler, initial_epoch)
    for loss, epoch in training_iter:
        checkpoint_num += 1

        top5_accuracy, similarity = test(model.word2vec, test_samples)

        with open(config.logs_file, 'a') as file:
            file.write(f'{loss} {top5_accuracy} {similarity}\n')

        filename = f'checkpoint-{checkpoint_num}.pth'
        torch.save(dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            top5_accuracy=top5_accuracy,
            similarity=similarity,
            initial_epoch=epoch,
        ), os.path.join(config.checkpoints_dir, filename))
