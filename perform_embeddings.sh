#!/bin/bash

heat=/rds/projects/2018/hesz01/heat/heat/heat_embedding.py

for theta1 in {10..95..5}
do
	for ((theta2=5; theta2<$theta1; theta2+=5))
	do
		for seed in {0..29}
		do
			parameters=$(printf "theta1=0.%02d-theta2=0.%02d-seed=%02d" ${theta1} ${theta2} ${seed})
			edgelist_dir=edgelists/synthetic_bow_tie/${dir}
			embedding_dir=embeddings/synthetic_bow_tie/${dir}
			walks_dir=walks/synthetic_bow_tie/${dir}

			slurm_options=$(echo \
			--job-name=performEmbeddings${parameters}\
			--time=03:00:00 \
			--mem=3G \
			--output=performEmbeddings${parameters}.out \
			--error=performEmbeddings${parameters}.err
			)

			modules=$(echo \
			module purge\; \
			module load bluebear\; \
			module load apps/python3/3.5.2\; \
			module load apps/keras/2.0.8-python-3.5.2
			)

			cmd=$(echo python ${heat} --edgelist ${edgelist_dir}.edgelist --features none --labels ${edgelist_dir}.csv \
			--embedding ${embedding_dir} --walks ${walks_dir} --dim 10 -b 50 -e 25 --sigma 3 --context-size 1 --directed)

			sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd})

		done
	done
done
