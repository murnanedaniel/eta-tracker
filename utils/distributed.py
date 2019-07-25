"""Utility code for running distributed"""

import os

from torch import nn
import torch.distributed as dist

def init_workers_file():
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = 'file:///tmp/%s_%s_pytorch_sync' % (
        os.environ['USER'], os.environ['SLURM_JOB_ID'])
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_mpi():
    rank, n_ranks = 0, 1
    dist.init_process_group(backend='mpi')
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    return rank, n_ranks

def init_workers_cray():
    import dl_comm.torch as cdl
    rank = cdl.get_rank()
    n_ranks = cdl.get_nranks()
    return rank, n_ranks

def distribute_model(model, mode=None, gpu=None):
    # PyTorch distributed for GPUs using file initialization
    if mode == 'ddp-file':
        return nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], output_device=gpu)
    # CPU + MPI only
    elif mode == 'ddp-mpi':
        return nn.parallel.DistributedDataParallelCPU(model)
    # With cray plugin we instead wrap the optimizer, not the model
    elif mode == 'cray':
        pass
    return model

def distribute_optimizer(optimizer, mode=None):
    if mode == 'cray':
        # Wrap the optimizer in order to use the Plugin's communication.
        import dl_comm.torch as cdl
        nteam = 1 # number of teams you'll be training
        nthread = 2 # number of communication threads
        optimizer = cdl.DistributedOptimizer(optimizer, nteam=nteam,
                                             nthread_per_team=nthread)
    return optimizer
