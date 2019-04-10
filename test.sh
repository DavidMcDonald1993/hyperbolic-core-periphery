#!/bin/bash

slurm_options=$(echo \
--qos=bbshort \
--job-name=testJob \
--time=01:00:00 \
--mem=1G
)

modules=$(echo \
module purge\; \
module load bluebear\; \
module load apps/python3/3.5.2\;
)

cmd="echo \"hello world\""

echo sbatch ${slurm_options} \"${modules} ${cmd}\"
sbatch ${slurm_options} \"${modules} ${cmd}\"