"""Utility code for running distributed"""

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
    # FINISH ME
    return 1, 0

def init_workers_cray():
    # FINISH ME
    return 1, 0

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
        pass
    return optimizer
