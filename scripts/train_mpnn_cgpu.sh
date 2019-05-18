#!/bin/bash
#SBATCH -J mpnn-big-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/mpnn_big.yaml
mkdir -p logs
. scripts/setup_cgpu.sh

# Single GPU training
srun -u python train.py $config --gpu 0 $@
