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
from datasets.graph import Graph, save_graph
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


def select_hits():
    """ Future-proofing: May select triplets based on angle between doublets"""
    pass

def load_pid(filename):
    return np.load(filename)["pid"]

def load_doublet_data():

    input_dir = '/mnt/c/Users/Daniel/Dropbox/Research/Publications/ExaTrkX/data/hitgraphs_tiny'
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.startswith('event') and not f.endswith('_ID.npz') and not f.endswith('_pid.npz')]
    doublet_data = graph.load_graphs(filenames)

    pid_filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.startswith('event') and f.endswith('_pid.npz')]

    pid_data = [load_pid(f) for f in pid_filenames]

    return doublet_data, pid_data

def get_edge_scores(result_dir, n_graphs):
    """
    - Takes config info for triplet training dataset (different from doublet training dataset),
    - Runs the dataset through the trained doublet network,
    - Returns edge scores with same indices as edge network input
    """

    # Load configs
    config = load_config_dir(result_dir)
    logging.info('Training doublets on model configuration:')
    logging.info(config)

    # Find the best epoch
    summaries = load_summaries(config)
    best_idx = summaries.valid_loss.idxmin()
    summaries.loc[[best_idx]]

    # Build the trainer and load best checkpoint
    trainer = get_trainer(output_dir=config['output_dir'], **config['trainer'])
    trainer.build_model(optimizer_config=config['optimizer'], **config['model'])

    best_epoch = summaries.epoch.loc[best_idx]
    trainer.load_checkpoint(checkpoint_id=best_epoch)

    logging.info("With weight system:")
    logging.info(trainer.model)
    # print('Parameters:', sum(p.numel() for p in trainer.model.parameters()))

    # Load the test dataset

    test_loader = get_test_data_loader(config, n_test=n_graphs)
    # Apply the model
    test_preds, test_targets = trainer.predict(test_loader)
    doublet_data = test_loader.dataset

    return test_preds, doublet_data


def edge_to_triplet(start, end, n_edges, n_hits):
    """
    An efficient algorithm to convert between an edge matrix and a triplet matrix
    """
    Ri = np.zeros((n_hits+1, n_edges))
    Ro = np.zeros((n_hits+1, n_edges))
    Ri[start, np.arange(n_edges)]=1
    Ro[end, np.arange(n_edges)]=1
    Riwhere = [np.nonzero(t)[0] for t in Ri]
    Rowhere = [np.nonzero(t)[0] for t in Ro]
    E = [np.stack(np.meshgrid(j, i),-1).reshape(-1,2) for i,j in zip(Riwhere, Rowhere)]
    return np.concatenate(E).T

def construct_triplet_graph(x,e,pid,o):
    """
    Very similar to doublet graph builder. May take some pruning parameters.
    Takes output from doublet network.
    """

    # Initialise useful values
    start, end = e
    n_edges = len(start)
    n_hits = np.max(e)

    # Build triplet edge index matrix
    triplet_index = edge_to_triplet(start, end, n_edges, n_hits)
    n_triplets = triplet_index.shape[1]

    # Concatenate features by edge index
    triplet_X = np.concatenate([x[e[0]],x[e[1]],np.array([o]).T], axis=1)

    # Ground truth vector from THREE matching pids in the triplet edge
    triplet_y = np.zeros(n_triplets)
    triplet_y[:] = int((pid[triplet_index[0]] == pid[triplet_index[1]]) and (pid[triplet_index[0]] != 0))

    # Convert the triplet_index matrix back to association matrices
    # NOTE: This is an inefficient process, since this is converted back later...
    triplet_Ri = np.zeros((n_edges, n_triplets), dtype=np.uint8)
    triplet_Ro = np.zeros((n_edges, n_triplets), dtype=np.uint8)
    triplet_Ri[triplet_index[0], np.arange(n_triplets)] = 1
    triplet_Ro[triplet_index[1], np.arange(n_triplets)] = 1


    return Graph(triplet_X, triplet_Ri, triplet_Ro, triplet_y)
    # return SparseGraph(X, edge_index, y)


def process_event(data_row, output_dir):
    """ Handles all events, returns nothing. As in doublet case"""

    # doublet_data, pid_data = load_doublet_data()

#     for gi, oi, i in zip(doublet_data, edge_scores, np.arange(len(doublet_data))):
#         x, e, pid, o = gi.x.numpy(), gi.edge_index.numpy(), gi.pid.numpy(), oi.numpy() # Divide out feature_scale???
#         logging.info("Constructing graph " + str(i))
#         graphs_all.append(construct_triplet_graph(x,e,pid,o))

    x, e, pid, o, filename = data_row

    graph = construct_triplet_graph(x, e, pid, o)

    logging.info("Constructing graph " + str(filename))

    save_graph(graph, os.path.join(output_dir, 'g_%03i' % filename))
#     p = mp.Pool(processes=n_workers)
#     p.imap(save_graph_map, zip(graphs_all, ))
#     p.close()
#     p.join()

#     save_graphs(graphs_all, filenames)


def process_data(output_dir, result_dir, n_files, n_workers):

    logging.info("Processing result data")

    edge_scores, doublet_data = get_edge_scores(result_dir, n_files)
    all_data = np.array([[gi.x.numpy(), gi.edge_index.numpy(), gi.pid.numpy(), oi.numpy()]
                    for gi, oi in zip(doublet_data, edge_scores)])
    all_data = np.c_[all_data, np.arange(len(all_data)).T]
    print(all_data.shape)
    logging.info("Data processed")

    with mp.Pool(processes=n_workers) as pool:
        process_fn = partial(process_event, output_dir=output_dir)
        pool.map(process_fn, all_data)


def main():
    """ Main function """

    # Parse args
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

    result_dir = config['doublet_model_dir']
    output_dir = config['output_dir']

    process_data(output_dir, result_dir, config['n_graphs'], args.n_workers)

#     process_events(output_dir, result_dir, config['n_graphs'], args.n_workers)

    logging.info('Processing finished')


if __name__ == '__main__':
    main()
