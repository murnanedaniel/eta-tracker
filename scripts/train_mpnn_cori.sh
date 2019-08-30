#!/bin/bash
#SBATCH -J mpnn-cori
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/mpnn.yaml
mkdir -p logs
. scripts/setup_cori.sh

srun -u python train.py $config $@
