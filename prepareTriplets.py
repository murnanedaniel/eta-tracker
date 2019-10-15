"""
THIS PREPARATION SCRIPT IS TO GET A DATASET READY FOR TRIPLET TRAINING
"""

# System
import os
import sys
from pprint import pprint as pp
import argparse
import logging
import multiprocessing as mp
from functools import partial

# Externals
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_scatter import scatter_add
import trackml.dataset

# Locals
from datasets.graph import Graph, save_graphs
from datasets import get_data_loaders
from trainers import get_trainer
from datasets import graph
from notebooks.nb_utils import (load_config_file, load_config_dir, load_summaries,
                      plot_train_history, get_test_data_loader,
                      compute_metrics, plot_metrics, draw_sample_xy)

# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser('prepare.py')
#     add_arg = parser.add_argument
#     add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
#     add_arg('--n-workers', type=int, default=1)
#     add_arg('--task', type=int, default=0)
#     add_arg('--n-tasks', type=int, default=1)
#     add_arg('-v', '--verbose', action='store_true')
#     add_arg('--show-config', action='store_true')
#     add_arg('--interactive', action='store_true')
#     return parser.parse_args()
#

"""
Possibly some pruning scripts
"""

""" OVERALL STRUCTURE
Doublet preparation script makes graphs of (X, Ri, Ro, y, pid) -> Doublet trainer loads graphs with (X, edge_index, y)
-> Doublets trained on (X, e, y) -> Triplet preparation reads (X, Ri, Ro, y) -> Runs through doublet classifier
-> Makes graph of ([Xi,Xo,edge_score], triplet_e, y) and saves -> Triplet trainer loads graph -> Triplets trained
"""

def load_pid(filename):
    return np.load(filename)["pid"]

def get_edge_scores():
    """
    - Takes config info for triplet training dataset (different from doublet training dataset),
    - Runs the dataset through the trained doublet network,
    - Returns edge scores with same indices as edge network input
    """

    # Load by config file
    config_file = 'configs/tripgnn.yaml'
    config = load_config_file(config_file)
    summaries = load_summaries(config)

    # Load by directory (preferred)
    result_base = os.path.expandvars('/mnt/c/Users/Daniel/Dropbox/Research/Publications/ExaTrkX/results')

    result_name = 'agnn01'
    result_dir = os.path.join(result_base, result_name)

    config = load_config_dir(result_dir)
    print('Configuration:')
    pp(config)

    summaries = load_summaries(config)
    print('\nTraining summaries:')
    pp(summaries)

    # Find the best epoch
    best_idx = summaries.valid_loss.idxmin()
    summaries.loc[[best_idx]]

    # Build the trainer and load best checkpoint
    trainer = get_trainer(output_dir=config['output_dir'], **config['trainer'])
    trainer.build_model(optimizer_config=config['optimizer'], **config['model'])

    best_epoch = summaries.epoch.loc[best_idx]
    trainer.load_checkpoint(checkpoint_id=best_epoch)

    print(trainer.model)
    print('Parameters:', sum(p.numel() for p in trainer.model.parameters()))

    # Load the test dataset
    n_test = 64
    test_loader = get_test_data_loader(config, n_test=n_test)
    # Apply the model
    test_preds, test_targets = trainer.predict(test_loader)

    return test_preds

def load_doublet_data():

    input_dir = '/mnt/c/Users/Daniel/Dropbox/Research/Publications/ExaTrkX/data/hitgraphs_small_000'
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.startswith('event') and not f.endswith('_ID.npz') and not f.endswith('_pid.npz')]
    doublet_data = graph.load_graphs(filenames)

    pid_filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.startswith('event') and f.endswith('_pid.npz')]

    pid_data = [load_pid(f) for f in pid_filenames]
    pp(pid_data)

    return doublet_data, pid_data


def construct_triplet_graph():
    """
    Very similar to doublet graph builder. May take some pruning parameters.
    Will take output from doublet network, so should use numpy.
    - Takes feature list X, edge_index matrix, particle_id list (stored as
    sparse edge list [0,0,0,100,0,2,...] or densely {3:100,5:2,...}),
    - Concatenate Xi, Xo along edge_indices
    - Build triplet_edge_index matrix e
    - Build ground truth triplet_y
    - PRUNING?
    - Returns Graph([Xi,Xo], Ri/Ro, triplet_y)
    - NOTE: That means we use hitgraph_builder as in doublet case
    """
    pass


def select_hits():
    """ Future-proofing: May select triplets based on angle between doublets"""
    pass



def process_event():
    """ Handles an event, returns file dataset. As in doublet case"""
    pass


def main():
    """ Main function """

    # Parse args
    # args = parse_args()
    #
    # # Setup logging
    # log_format = '%(asctime)s %(levelname)s %(message)s'
    # log_level = logging.DEBUG if args.verbose else logging.INFO
    # logging.basicConfig(level=log_level, format=log_format)
    # logging.info('Initialising')
    # if args.show_config:
    #     logging.info('Command line config: %s' % config)
    #
    # # Load config
    # with open(args.config) as f:
    #     config = yaml.load(f)
    # if args.task == 0:
    #     logging.info('Configuration: %s' % config)
    #
    # input_dir = config['input_dir']
    # all_files = os.listdir(input_dir)
    # suffix = ''
    #
    # train_data_loader, valid_data_loader = get_data_loaders(
    #     distributed=is_distributed, rank=rank, n_ranks=n_ranks, **config['data'])
    # logging.info('Loaded %g training samples', len(train_data_loader.dataset))

    # edge_scores = get_edge_scores()
    doublet_data, pid_data = load_doublet_data()



if __name__ == '__main__':
    main()
