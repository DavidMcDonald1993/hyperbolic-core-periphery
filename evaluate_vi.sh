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
module load apps/python3/3.5.2
module load apps/scikit-learn/0.19.0-python-3.5.2

echo "starting "${ARR[${SLURM_ARRAY_TASK_ID}]}
python cluster/evaluate_vi.py ${ARR[${SLURM_ARRAY_TASK_ID}]}
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
