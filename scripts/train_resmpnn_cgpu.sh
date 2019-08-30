#!/bin/bash
#SBATCH -J resmpnn-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH -t 8:00:00
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/resmpnn.yaml
. scripts/setup_cgpu.sh
mkdir -p logs

# Single GPU training
srun -u python train.py $config --gpu 0 $@
