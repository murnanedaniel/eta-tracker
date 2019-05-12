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

class GNNTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""

    def __init__(self, real_weight=1, fake_weight=1, **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def build_model(self, name='gnn_sparse',
                    loss_func='binary_cross_entropy_with_logits',
                    optimizer='Adam', learning_rate=0.001,
                    lr_scaling=None, lr_warmup_epochs=0,
                    lr_decay_schedule=[], **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(name=name, **model_args).to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[self.gpu], output_device=self.gpu)

        # Construct the loss function
        self.loss_func = getattr(nn.functional, loss_func)

        # Construct the optimizer
        if lr_scaling == 'linear':
            learning_rate = learning_rate * self.n_ranks
            warmup_factor = 1. / self.n_ranks
        elif lr_scaling == 'sqrt':
            learning_rate = learning_rate * math.sqrt(self.n_ranks)
            warmup_factor = 1. / math.sqrt(self.n_ranks)
        else:
            warmup_factor = 1
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(), lr=learning_rate)

        # LR schedule
        def lr_schedule(epoch, warmup_factor=warmup_factor,
                        warmup_epochs=lr_warmup_epochs,
                        decays=lr_decay_schedule):
            if epoch < warmup_epochs:
                return (1 - warmup_factor) * epoch / warmup_epochs + warmup_factor
            for decay in decays:
                if epoch >= decay['start_epoch'] and epoch < decay['end_epoch']:
                    return decay['factor']
            else:
                return 1

        # LR schedule
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_schedule)

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
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            # Compute target weights on-the-fly for loss function
            batch_weights_real = batch.y * self.real_weight
            batch_weights_fake = (1 - batch.y) * self.fake_weight
            batch_weights = batch_weights_real + batch_weights_fake
            self.model.zero_grad()
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y, weight=batch_weights)
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
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y).item()
            sum_loss += batch_loss
            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            matches = ((batch_pred > 0.5) == (batch.y > 0.5))
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
