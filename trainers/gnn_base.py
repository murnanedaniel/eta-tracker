"""
Common PyTorch trainer code.
"""

# System
import os
import re
import logging
import time
import math

# Externals
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Locals
from models import get_model
from utils.optimizers import get_lr_scheduler
from utils.distributed import distribute_model, distribute_optimizer

class GNNBaseTrainer(object):
    """
    Base class for GNN PyTorch trainers.
    This implements the common training logic,
    model construction and distributed training setup,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, gpu=None,
                 distributed_mode=None, rank=0, n_ranks=1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        self.gpu = gpu
        if gpu is not None:
            self.device = 'cuda:%i' % gpu
            torch.cuda.set_device(gpu)
        else:
            self.device = 'cpu'
        self.distributed_mode = distributed_mode
        self.summaries = None
        self.summary_file = None
        self.rank = rank
        self.n_ranks = n_ranks

    def _build_optimizer(self, parameters, name='Adam', learning_rate=0.001,
                         lr_scaling=None, n_ranks=1, **optimizer_args):
        """Construct the training optimizer and scale learning rate.
        Should be called by build_model rather than called directly."""

        # Scale the learning rate
        if lr_scaling == 'linear':
            learning_rate = learning_rate * n_ranks
        elif lr_scaling == 'sqrt':
            learning_rate = learning_rate * math.sqrt(n_ranks)

        # Construct the optimizer
        OptimizerType = getattr(torch.optim, name)
        optimizer = OptimizerType(parameters, lr=learning_rate, **optimizer_args)
        return optimizer

    def build_model(self, name='gnn_sparse',
                    loss_func='binary_cross_entropy_with_logits',
                    optimizer={}, lr_scaling='linear', lr_warmup_epochs=0, lr_decay_schedule=[],
                    **model_args):
        """Instantiate our model"""

        self.logger.info(optimizer)
        # Construct the model
        model = get_model(name=name, **model_args).to(self.device)
        self.model = distribute_model(model, mode=self.distributed_mode, gpu=self.gpu)

        # Construct the loss function
        self.loss_func = getattr(nn.functional, loss_func)

        # Construct the optimizer
        optimizer = self._build_optimizer(self.model.parameters(),
                                          n_ranks=self.n_ranks, lr_scaling=lr_scaling,
                                          **optimizer)
        self.optimizer = distribute_optimizer(optimizer, mode=self.distributed_mode)

        # LR schedule
        self.lr_scheduler = get_lr_scheduler(self.optimizer,
                                             lr_scaling=lr_scaling,
                                             n_ranks=self.n_ranks,
                                             warmup_epochs=lr_warmup_epochs,
                                             decay_schedule=lr_decay_schedule)

    def save_summary(self, summaries):
        """Save summary information.
        This implementation is currently inefficient for simplicity:
            - we build a new DataFrame each time
            - we write the whole summary file each time
        """
        if self.summaries is None:
            self.summaries = pd.DataFrame([summaries])
        else:
            self.summaries = self.summaries.append([summaries], ignore_index=True)
        if self.output_dir is not None:
            summary_file = os.path.join(self.output_dir, 'summaries_%i.csv' % self.rank)
            self.summaries.to_csv(summary_file, index=False)

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the model"""
        assert self.output_dir is not None
        checkpoint = dict(checkpoint_id=checkpoint_id,
                          model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          lr_scheduler=self.lr_scheduler.state_dict())
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_file))

    def load_checkpoint(self, checkpoint_id=-1):
        """Load a model checkpoint"""
        assert self.output_dir is not None

        # Load the summaries
        summary_file = os.path.join(self.output_dir, 'summaries_%i.csv' % self.rank)
        logging.info('Reloading summary at %s', summary_file)
        self.summaries = pd.read_csv(summary_file)

        # Load the checkpoint
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        if checkpoint_id == -1:
            # Find the last checkpoint
            last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
            pattern = 'model_checkpoint_(\d..).pth.tar'
            checkpoint_id = int(re.match(pattern, last_checkpoint).group(1))
        checkpoint_file = 'model_checkpoint_%03i.pth.tar' % checkpoint_id
        logging.info('Reloading checkpoint at %s', checkpoint_file)
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def print_model_summary(self):
        """Override as needed"""
        self.logger.info(
            'Model: \n%s\nParameters: %i' %
            (self.model, sum(p.numel() for p in self.model.parameters())))

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""

        # Determine initial epoch in case resuming training
        start_epoch = 0
        if self.summaries is not None:
            start_epoch = self.summaries.epoch.max() + 1

        # Loop over epochs
        for epoch in range(start_epoch, n_epochs):
            self.logger.info('Epoch %i' % epoch)
            try:
                train_data_loader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

            # Train on this epoch
            start_time = time.time()
            summary = self.train_epoch(train_data_loader)
            summary['epoch'] = epoch
            summary['train_time'] = time.time() - start_time

            # Evaluate on this epoch
            if valid_data_loader is not None:
                start_time = time.time()
                summary.update(self.evaluate(valid_data_loader))
                summary['valid_time'] = time.time() - start_time

            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None and self.rank == 0:
                self.write_checkpoint(checkpoint_id=epoch)

        return self.summaries
