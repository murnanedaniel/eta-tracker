#!/bin/bash
#SBATCH -J segclf-med
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out

# Disabled slurm config
###SBATCH --exclusive
###SBATCH --gres=gpu:8

config=configs/segclf_med.yaml

mkdir -p logs

# Temporarily use project instead of scratch
export SCRATCH=/project/projectdirs/nstaff/sfarrell

# Setup software
#conda activate /global/common/cori/software/pytorch/v1.0.0-gpu
module load pytorch/v1.0.0-gpu

# Setup MPI enabled software
#conda activate /project/projectdirs/nstaff/sfarrell/conda/pytorch/v1.0.1-gpu
#export MV2_ENABLE_AFFINITY=0

# 1 GPU
srun -u python train.py $config --gpu 0 $@

# 8 GPUs
#srun -n 8 -l -u python train.py $config -d --rank-gpu $@
