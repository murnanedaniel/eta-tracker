#!/bin/bash
#SBATCH -J mpnn-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH -t 8:00:00
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

# Multi-GPU settings
##SBATCH --gres=gpu:8
##SBATCH --exclusive

# Single-GPU settings
##SBATCH --gres=gpu:1
##SBATCH --mem=30GB

# Setup
config=configs/mpnn.yaml
mkdir -p logs
. scripts/setup_cgpu.sh

# Single GPU training
srun -u python train.py $config --gpu 0 $@

# Multi-GPU training
#srun -u -l --ntasks-per-node 8 \
#    python train.py $config --rank-gpu -d ddp-file $@
