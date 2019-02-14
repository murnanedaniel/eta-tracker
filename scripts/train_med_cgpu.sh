#!/bin/bash
#SBATCH -J segclf-med
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out

mkdir -p logs
export SCRATCH=/project/projectdirs/nstaff/sfarrell
conda activate /global/common/cori/software/pytorch/v1.0.0-gpu
#conda activate /project/projectdirs/nstaff/sfarrell/conda/pytorch/v1.0.1-gpu
#export MV2_ENABLE_AFFINITY=0

srun -n 8 -l -u python train.py configs/segclf_med.yaml -d --rank-gpu
