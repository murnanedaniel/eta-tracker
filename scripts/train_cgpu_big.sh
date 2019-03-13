#!/bin/bash
#SBATCH -J segclf-big
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -t 1:40:00
#SBATCH -o logs/%x-%j.out

config=configs/segclf_big.yaml
mkdir -p logs

# Setup software
module load pytorch/v1.0.0-gpu

# 1 GPU
srun -u python train.py $config --gpu 0 $@
