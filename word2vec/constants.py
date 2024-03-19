import os
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
PATH2DATA = r'/media/data/Datasets/AMAZON_FASHION.json'

EXTRA_DIR = 'extra'
TEST_FILE = os.path.join(EXTRA_DIR, 'test.txt')

SAVE_DIR = 'logs'
CHECKPOINTS_DIR = os.path.join(SAVE_DIR, 'checkpoints')
LOGS_FILE = os.path.join(SAVE_DIR, 'logs.txt')

UNK_TOKEN = '<unk>'
