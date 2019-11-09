"""
This module defines a generic trainer for simple models and datasets.
"""

# Externals
import torch
import numpy as np

# Locals
from .gnn_base import GNNBaseTrainer
from utils.checks import get_weight_norm

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0

        # Loop over training batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            if (i == 1) or (i==2):        
                print("Prediction: ", batch_output, " location: ", torch.argmax(batch_output))
                print("Truth: ", torch.argmax(batch.y))
            # This converts the one_hot vector to an integer and calculates the loss against it
            batch_loss = self.loss_func(batch_output.unsqueeze(0), torch.argmax(batch.y).unsqueeze(0))
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            self.logger.debug('  train batch %i, loss %f', i, batch_loss.item())

        # Summarize the epoch
        n_batches = i + 1
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
#             print("i: ", i, " batch: ", batch)
            # Make predictions on this batch
#             print(batch)
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output.unsqueeze(0), torch.argmax(batch.y).unsqueeze(0))
            sum_loss += batch_loss

            # Count number of correct predictions
            m = torch.nn.Softmax()
            batch_pred = m(batch_output)
            match = torch.argmax(batch_pred) == torch.argmax(batch.y)
#             print("Prediction: ", batch_pred, " location: ", torch.argmax(batch_pred))
#             print("Truth: ", torch.argmax(batch.y))
            sum_correct += int(match)
#             sum_total += matches.numel()
#             self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / n_batches
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

    @torch.no_grad()
    def predict(self, data_loader):
        preds, targets = [], []
        for batch in data_loader:
            preds.append(torch.sigmoid(self.model(batch)).squeeze(0))
            targets.append(batch.y.squeeze(0))
        return preds, targets

def _test():
    t = SparseGNNTrainer(output_dir='./')
    t.build_model()
