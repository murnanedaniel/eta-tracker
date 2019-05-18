#!/bin/bash
#SBATCH -J resmpnn-big-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/resmpnn_big.yaml
. scripts/setup_cgpu.sh
mkdir -p logs

# Single GPU training
srun -u python train.py $config --gpu 0 $@
