#!/bin/bash
#SBATCH -J agnn-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/agnn.yaml
mkdir -p logs
. scripts/setup_cgpu.sh

# Single GPU training
#srun -u python train.py $config --gpu 0 $@

# Multi-GPU training with MPI
srun -u -l --ntasks-per-node 8 \
    python train.py $config --rank-gpu -d ddp-mpi $@
