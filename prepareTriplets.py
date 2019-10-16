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
from datasets.graph import Graph, save_graphs, save_graph
from notebooks.nb_utils import (load_config_file, load_config_dir, load_summaries,
                      plot_train_history, get_test_data_loader,
                      compute_metrics, plot_metrics, draw_sample_xy)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()


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

def get_edge_scores(result_name):
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
    n_test = 10
    test_loader = get_test_data_loader(config, n_test=n_test)
    # Apply the model
    test_preds, test_targets = trainer.predict(test_loader)
    doublet_data = test_loader.dataset

    return test_preds, doublet_data

def load_doublet_data():

    input_dir = '/mnt/c/Users/Daniel/Dropbox/Research/Publications/ExaTrkX/data/hitgraphs_tiny'
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.startswith('event') and not f.endswith('_ID.npz') and not f.endswith('_pid.npz')]
    doublet_data = graph.load_graphs(filenames)

    pid_filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.startswith('event') and f.endswith('_pid.npz')]

    pid_data = [load_pid(f) for f in pid_filenames]

    return doublet_data, pid_data

def edge_to_triplet(start, end, n_edges, n_hits):
    Ri = np.zeros((n_hits+1, n_edges))
    Ro = np.zeros((n_hits+1, n_edges))
    Ri[start, np.arange(n_edges)]=1
    Ro[end, np.arange(n_edges)]=1
    Riwhere = [np.nonzero(t)[0] for t in Ri]
    Rowhere = [np.nonzero(t)[0] for t in Ro]
    Riwhere, Rowhere
    E = [np.stack(np.meshgrid(j, i),-1).reshape(-1,2) for i,j in zip(Riwhere, Rowhere)]
    return np.concatenate(E).T

def construct_triplet_graph(x,e,pid,o):
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

    start, end = e
    n_edges = len(start)
    n_hits = np.max(e)
    triplet_index = edge_to_triplet(start, end, n_edges, n_hits)
    n_triplets = triplet_index.shape[1]
    triplet_X = np.concatenate([x[e[0]],x[e[1]],np.array([o]).T], axis=1)
    triplet_y = np.zeros(n_triplets, dtype=np.float32)
    triplet_y = pid[triplet_index[0]] == pid[triplet_index[1]]
    # triplet_pid = triplet_y*pid[triplet_index[0]]
    triplet_Ri = np.zeros((n_edges, n_triplets), dtype=np.uint8)
    triplet_Ro = np.zeros((n_edges, n_triplets), dtype=np.uint8)
    triplet_Ri[triplet_index[0], np.arange(n_triplets)] = 1
    triplet_Ro[triplet_index[1], np.arange(n_triplets)] = 1

    return Graph(triplet_X, triplet_Ri, triplet_Ro, triplet_y)
    # return SparseGraph(X, edge_index, y)


def select_hits():
    """ Future-proofing: May select triplets based on angle between doublets"""
    pass



def process_events(prefix, output_dir):
    """ Handles all events, returns nothing. As in doublet case"""

    # doublet_data, pid_data = load_doublet_data()
    edge_scores, doublet_data = get_edge_scores(doublet_filenames)
    print("All data loaded")
    graphs_all = []

    for gi, oi, i in zip(doublet_data, edge_scores, np.arange(len(doublet_data))):
        x, e, pid, o = gi.x.numpy(), gi.edge_index.numpy(), gi.pid.numpy(), oi.numpy() # Divide out feature_scale???
        print("Constructing graph " + str(i) + " in file " + result_name + "i")
        all_graphs.append(construct_triplet_graph(x,e,pid,o))

    try:
        base_prefix = os.path.basename(prefix)
        filenames = [os.path.join(output_dir, '%s_g%03i' % (base_prefix, i))
                     for i in range(len(graphs_all))]

    save_graphs(all_graphs, filenames)

    # """ List comprehension would be nicer... """
    # all_graphs = [construct_triplet_graph(gi.x.numpy(), gi.edge_index.numpy(), gi.pid.numpy(), oi.numpy())
    #                 for  gi, oi in zip(doublet_data, edge_scores)]


def main():
    """ Main function """

    Parse args
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initialising')
    if args.show_config:
        logging.info('Command line config: %s' % config)

    # Load config
    with open(args.config) as f:
        config = yaml.load(f)
    if args.task == 0:
        logging.info('Configuration: %s' % config)

    input_dir = config['input_dir']
    all_files = os.listdir(input_dir)
    suffix = ''

    # train_data_loader, valid_data_loader = get_data_loaders(
    #     distributed=is_distributed, rank=rank, n_ranks=n_ranks, **config['data'])
    # logging.info('Loaded %g training samples', len(train_data_loader.dataset))

    # edge_scores = get_edge_scores()

    doublet_filenames = "agnn01"
    output_filenames =

    process_events(result_name, )


if __name__ == '__main__':
    main()
