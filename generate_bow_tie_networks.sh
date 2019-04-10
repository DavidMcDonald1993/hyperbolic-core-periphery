#!/bin/bash

#SBATCH --job-name=generateBowTieNetworks
#SBATCH --output=generateBowTieNetworks_%A_%a.out
#SBATCH --error=generateBowTieNetworks_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=3G
# SBATCH --mail-type ALL

ARR=("--seed="{0..29}})

module purge; module load bluebear
module load apps/python3/3.5.2

echo "starting "${ARR[${SLURM_ARRAY_TASK_ID}]}
python src/synthetic_bow_tie_generation.py ${ARR[${SLURM_ARRAY_TASK_ID}]}
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
