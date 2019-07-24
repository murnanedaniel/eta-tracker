#!/bin/bash
#SBATCH -J agnn-big
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# Setup
config=configs/agnn_big.yaml
mkdir -p logs
. scripts/setup_cori.sh

# Run training
srun -l -u python train.py $config $@
