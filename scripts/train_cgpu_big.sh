#!/bin/bash
#SBATCH -J segclf-big
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out

config=configs/segclf_big.yaml
mkdir -p logs

# Temporarily use project instead of scratch
export SCRATCH=/project/projectdirs/nstaff/sfarrell

# Setup software
module load pytorch/v1.0.0-gpu

# 1 GPU
srun -u python train.py $config --gpu 0 $@
