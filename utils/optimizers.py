"""Optimizer utility code"""

import match
from functools import partial

import torch

def get_optimizer(optimizer, parameters, learning_rate,
                  lr_scaling=None, n_ranks=1):
    # Scale the learning rate
    if lr_scaling == 'linear':
        learning_rate = learning_rate * n_ranks
    elif lr_scaling == 'sqrt':
        learning_rate = learning_rate * math.sqrt(n_ranks)
    return getattr(torch.optim, optimizer)(parameters, lr=learning_rate)

def _lr_schedule(epoch, warmup_factor=1, warmup_epochs=0, decays=[]):
    if epoch < warmup_epochs:
        return (1 - warmup_factor) * epoch / warmup_epochs + warmup_factor
    for decay in decays:
        if epoch >= decay['start_epoch'] and epoch < decay['end_epoch']:
            return decay['factor']
    else:
        return 1

def get_lr_scheduler(optimizer, lr_scaling=None, n_ranks=1, warmup_epochs=0,
                     decay_schedule=[]):
    warmup_factor = 1
    if lr_scaling == 'linear':
        warmup_factor = 1. / n_ranks
    elif lr_scaling == 'sqrt':
        warmup_factor = 1. / math.sqrt(n_ranks)

    schedule = partial(_lr_scheduler, warmup_factor=warmup_factor,
                       warmup_epochs=warmup_epochs, decays=decay_schedule)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
