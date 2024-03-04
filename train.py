from word2vec import *

import torch
import torch.optim as optim

import os
import sys
import time

from tqdm import trange
from tqdm.notebook import trange as trange_colab

from typing import Generator, Tuple


def train(model: Word2Vec,
          ds: DataSet,
          optimizer: optim.Adam,
          scheduler: optim.lr_scheduler.StepLR,
          initial_epoch: int = 1,
          initial_sentence_num: int = 0
          ) -> Generator[Tuple[torch.Tensor, int, int], None, None]:
    assert 1 <= initial_epoch <= EPOCHS

    starting_point = time.time()
    losses = list()

    for epoch in range(initial_epoch, EPOCHS + 1):

        iterator = (trange, trange_colab)['ipykernel_launcher' in sys.argv[0]]
        for sentence_num in iterator(initial_sentence_num, len(ds), desc=f'Sentences', leave=False):
            for sample in ds[sentence_num]:

                optimizer.zero_grad()

                x_index, targets_indices, negative_indices = sample
                predictions = model(x_index)

                positive = torch.log(predictions[targets_indices]).sum()
                negative = torch.log(1 - predictions[negative_indices]).sum()
                loss = - positive - negative

                loss.backward()
                optimizer.step()

                losses.append(loss)
            else:
                scheduler.step()

            if time.time() - starting_point >= LOG_FREQ:
                yield sum(losses) / len(losses), epoch, sentence_num
                starting_point = time.time()
                losses = list()

        initial_sentence_num = 0
        ds.shuffle(epoch + 1)


def main():
    indices = [int(f[11:-4]) for f in os.listdir(CHECKPOINTS_DIR)] + [-1, 0]
    checkpoint_num = max(indices)

    if not checkpoint_num:
        initial_epoch = 1
        initial_sentence_num = 0

        model = Word2Vec().to(DEVICE)
        ds = DataSet(PATH2DATA, initial_seed=initial_epoch)

        optimizer = optim.Adam(model.parameters(), LR)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, STEP_SIZE,
            gamma=(TARGET_LR / LR) ** (STEP_SIZE / (EPOCHS * len(ds)))
        )
    else:
        filename = f'checkpoint-{checkpoint_num}.pth'
        checkpoint_dict = torch.load(os.path.join(CHECKPOINTS_DIR, filename), map_location=DEVICE)

        initial_epoch = checkpoint_dict['initial_epoch']
        initial_sentence_num = checkpoint_dict['initial_sentence_num']

        model = checkpoint_dict['model']
        ds = DataSet(PATH2DATA, initial_seed=checkpoint_dict['initial_epoch'])

        optimizer = checkpoint_dict['optimizer']
        scheduler = checkpoint_dict['scheduler']

    model.train()

    training_iter = train(model, ds, optimizer, scheduler, initial_epoch, initial_sentence_num)
    for loss, epoch, sentence_num in training_iter:
        checkpoint_num += 1

        with open(LOSSES_FILE, 'a') as file:
            file.write(f'{loss.item()}\n')

        filename = f'checkpoint-{checkpoint_num}.pth'
        torch.save({
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'loss': loss,
            'initial_epoch': epoch,
            'initial_sentence_num': sentence_num + 1
        }, os.path.join(CHECKPOINTS_DIR, filename))

    model.eval()

    torch.save(model.state_dict(), 'model.pth')
    vector = Vector(model.embedding, ds.vocabulary)
    torch.save(vector, 'vector.pth')


if __name__ == '__main__':
    main()
