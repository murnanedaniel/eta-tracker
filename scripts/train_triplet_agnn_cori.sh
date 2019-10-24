#!/bin/bash
#SBATCH -J triplet-agnn-cori
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/tripgnn.yaml
mkdir -p logs
. scripts/setup_cori.sh

# Run training
srun -l -u python train.py $config $@
