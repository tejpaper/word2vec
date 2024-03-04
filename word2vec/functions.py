import torch.nn as nn


def requires_grad(model: nn.Module, flag: bool):
    for param in model.parameters():
        param.requires_grad = flag
