#!/bin/bash
#SBATCH -J trip-prep-med
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

mkdir -p logs
. scripts/setup_cori.sh
config=configs/prep_tripgnn.yaml

echo $SLURM_JOB_NUM_NODES

# Loop over tasks (1 per node) and submit
i=0
while [ $i -lt $SLURM_JOB_NUM_NODES ]; do
    echo "Launching task $i"
    srun -N 1 python prepareTriplets.py \
        --n-workers 32 --task $i --n-tasks $SLURM_JOB_NUM_NODES $config &
    let i=i+1
done
wait
