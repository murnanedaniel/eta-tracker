"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import math

# Externals
import torch
from torch import nn

# Locals
from .base_trainer import BaseTrainer
from models import get_model
from utils.optimizers import get_optimizer, get_lr_scheduler
from utils.distributed import distribute_mode, distribute_optimizer

class GNNTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""

    def __init__(self, real_weight=1, fake_weight=1, **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def build_model(self, name='gnn_segment_classifier',
                    loss_func='binary_cross_entropy_with_logits',
                    optimizer='Adam', learning_rate=0.001,
                    lr_scaling=None, lr_warmup_epochs=0,
                    lr_decay_schedule=[], **model_args):
        """Instantiate our model"""

        # Construct the model
        model = get_model(name=name, **model_args).to(self.device)
        self.model = distribute_model(model, mode=self.distributed_mode, gpu=self.gpu)

        # Construct the loss function
        self.loss_func = getattr(nn.functional, loss_func)

        # Construct the optimizer
        optimizer = get_optimizer(optimizer, self.model.parameters(),
                                  learning_rate=learning_rate,
                                  lr_scaling=lr_scaling,
                                  n_ranks=self.n_ranks)
        self.optimizer = distribute_optimizer(optimizer, mode=self.distributed_mode)

        # LR schedule
        self.lr_scheduler = get_lr_scheduler(self.optimizer,
                                             lr_scaling=lr_scaling,
                                             n_ranks=self.n_ranks,
                                             warmup_epochs=lr_warmup_epochs,
                                             decay_schedule=lr_decay_schedule)

    def write_checkpoint(self, checkpoint_id):
        super(GNNTrainer, self).write_checkpoint(
            checkpoint_id=checkpoint_id,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict())

    def load_checkpoint(self, checkpoint_id=-1):
        checkpoint = super(GNNTrainer, self).load_checkpoint(checkpoint_id)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        summary = dict()
        sum_loss = 0
        self.lr_scheduler.step()
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            # Compute target weights on-the-fly for loss function
            batch_weights_real = batch_target * self.real_weight
            batch_weights_fake = (1 - batch_target) * self.fake_weight
            batch_weights = batch_weights_real + batch_weights_fake
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target, weight=batch_weights)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            self.logger.debug('  train batch %i, loss %f', i, batch_loss.item())

        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.debug(' Processed %i batches', i + 1)
        self.logger.debug(' Current LR %f', summary['lr'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target).item()
            sum_loss += batch_loss
            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            matches = ((batch_pred > 0.5) == (batch_target > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / sum_total
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
