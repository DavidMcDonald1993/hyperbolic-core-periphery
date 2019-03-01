#!/bin/bash

#SBATCH --job-name=generateNetworks
#SBATCH --output=generateNetworks_%A_%a.out
#SBATCH --error=generateNetworks_%A_%a.err
#SBATCH --array=0-359
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=3G
# SBATCH --mail-type ALL

# DATA_DIR="/rds/homes/d/dxm237/data"

ARR=("--seed="{0..29}" --exp="{one_core,two_core,two_core_with_residual}" --algorithm="{BE,divisive,KM_ER,KM_config})

module purge; module load bluebear
module load apps/python3/3.5.2

# pip install --user cpalgorithm

echo "starting "${ARR[${SLURM_ARRAY_TASK_ID}]}
python src/synthetic_network_generation.py ${ARR[${SLURM_ARRAY_TASK_ID}]}
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
