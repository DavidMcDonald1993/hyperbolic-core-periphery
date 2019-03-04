#!/bin/bash

#SBATCH --job-name=evaluateVI
#SBATCH --output=evaluateVI_%A_%a.out
#SBATCH --error=evaluateVI_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=3G
# SBATCH --mail-type ALL

# DATA_DIR="/rds/homes/d/dxm237/data"

ARR=("--exp="{one_core,two_core,two_core_with_residual}" --algorithm="{BE,divisive,km_ER,km_config})

module purge; module load bluebear
module load Python/3.6.3-iomkl-2018a

# pip install --user cpalgorithm

echo "starting "${ARR[${SLURM_ARRAY_TASK_ID}]}
python src/evaluate_vi.py ${ARR[${SLURM_ARRAY_TASK_ID}]}
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
