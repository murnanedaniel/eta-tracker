#!/bin/bash
#SBATCH -J mpnn-big
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/mpnn_big.yaml
mkdir -p logs
. scripts/setup_cori.sh

srun -u python train.py configs/mpnn_big.yaml $@
