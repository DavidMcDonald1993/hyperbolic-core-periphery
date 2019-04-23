#!/bin/bash

hrs=10
mem=5G

heat=/rds/projects/2018/hesz01/heat/main.py


edgelist=edgelists/grn/edgelist.tsv
embedding_dir=embeddings/grn/
walks_dir=walks/grn/

slurm_options=$(echo \
--job-name=performEmbeddingsGRN\
--time=${hrs}:00:00 \
--mem=${mem} \
--output=performEmbeddingsGRN.out \
--error=performEmbeddingsGRN.err
)

modules=$(echo \
module purge\; \
module load bluebear\; \
module load apps/python3/3.5.2\; \
module load apps/keras/2.0.8-python-3.5.2
)

cmd=$(echo python ${heat} --edgelist ${edgelist} \
--embedding ${embedding_dir} --walks ${walks_dir} --dim 10 -e 25 --sigma 1 --context-size 3 --directed --lr 3.)

echo submitting ${cmd}
sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd})
