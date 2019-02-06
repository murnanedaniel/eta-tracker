"""
Hello world PyTorch trainer.
"""

# System
import os

# Externals
import pandas as pd

# Locals
from .base_trainer import BaseTrainer

class HelloTrainer(BaseTrainer):
    """Hello world trainer object"""

    def __init__(self, **kwargs):
        super(HelloTrainer, self).__init__(**kwargs)

    def build_model(self):
        pass
    def write_checkpoint(self, checkpoint_id):
        pass
    def write_summaries(self):
        pass

    def load_checkpoint(self, checkpoint_id=-1):
        """Load a model checkpoint"""
        assert self.output_dir is not None
        summary_file = os.path.join(self.output_dir, 'summaries.csv')
        self.summaries = pd.read_csv(summary_file)

    def print_model_summary(self):
        self.logger.info('Hello world')

    def train_epoch(self, data_loader):
        summary = dict()
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.logger.info('  Train batch input %s, target %s',
                             batch_input.size(), batch_target.size())
        self.logger.info('  Processed %i training batches' % (i + 1))
        summary['train_loss'] = 0
        return summary

    def evaluate(self, data_loader):
        """"Evaluate the model"""
        summary = dict()
        # Loop over validation batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.logger.info('  Valid batch input %s, target %s',
                             batch_input.size(), batch_target.size())
        self.logger.info('  Processed %i validation batches' % (i + 1))
        summary['valid_loss'] = 0
        summary['valid_acc'] = 1
        return summary
