import os

import torch

DEVICE = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])

path2dataset = r'D:\App\Datasets\AmazonReviewData'
PATH2DATA = os.path.join(path2dataset, 'AMAZON_FASHION.json')

PATH2CLUSTERS = 'clusters.json'

SAVE_DIR = 'logs'
CHECKPOINTS_DIR = os.path.join(SAVE_DIR, 'checkpoints')
LOSSES_FILE = os.path.join(SAVE_DIR, 'losses.txt')

RANDOM_SEED = 42

SKIP_GRAM_SIZE = 5
NEGATIVE_S_NUM = 5
VOCAB_SIZE = 4000
UNK_TOKEN = '<unk>'

VECTOR_SIZE = 300
MAX_NORM = 1

LR = 0.025
TARGET_LR = 1e-8
STEP_SIZE = 1000
EPOCHS = 5
LOG_FREQ = 450
