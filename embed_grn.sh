#!/bin/bash

days=3
hrs=00
mem=5G

heat=/rds/projects/2018/hesz01/heat/main.py

edgelist=edgelists/grn/edgelist.tsv
embedding_dir=embeddings/grn/
walks_dir=walks/grn/

for dim in 05 10 25 50 100
do
	for seed in {00..29}
	do

		slurm_options=$(echo \
		--job-name=performEmbeddingsGRN-${dim}-${seed} \
		--time=${days}-${hrs}:00:00 \
		--mem=${mem} \
		--output=performEmbeddingsGRN-${dim}-${seed}.out \
		--error=performEmbeddingsGRN-${dim}-${seed}.err
		)

		modules=$(echo \
		module purge\; \
		module load bluebear\; \
		module load apps/python3/3.5.2\; \
		module load apps/keras/2.0.8-python-3.5.2\; \
		)

		cmd=$(echo python ${heat} --edgelist ${edgelist} \
		--embedding ${embedding_dir} --walks ${walks_dir} --dim ${dim} -e 25 --directed --seed ${seed} )

		echo submitting ${cmd}
		sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd})

	done
done



