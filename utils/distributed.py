"""Utility code for running distributed"""

from torch import nn
import torch.distributed as dist
import ml_comm_torch as cdl

def init_workers_file():
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = 'file:///tmp/%s_%s_pytorch_sync' % (
        os.environ['USER'], os.environ['SLURM_JOB_ID'])
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_mpi():
    # FINISH ME
    return 1, 0

def init_workers_cray():
    cdl.init_mpi()
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
    elif mode == 'cray':
        pass
    return model

def distribute_optimizer(optimizer, mode=None):
    # FINISH ME (for cray plugin)
    if mode == 'cray':
        # Wrap the optimizer in order to use the 
        # Plugin's communication. It's
        # completed as part of the base optimizer's step() method.
        # nsteps = len(train_sampler) # Number of steps training will go on for
        # TODO compute these automatically
        nsteps = 32768 # Number of steps training will go on for
        nteams = 1 # number of teams you'll be training
        nthreads = 2 # number of communication threads
        warmup = 0.10 #warm up first 10% of training
        verb = 2 # maximum verbosity
        freq = 1 # number of steps before outputing verbosity output
        optimizer = cdl.DistributedOptimizer(optimizer, nsteps, 
                        nteams, nthreads, warmup, verb, freq)
    return optimizer
