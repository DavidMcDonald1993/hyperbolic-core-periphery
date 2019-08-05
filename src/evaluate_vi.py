import sys
if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")
import os 

import numpy as np
import scipy as sp
import pandas as pd

from collections import Counter
from itertools import count
from collections import defaultdict as ddict

from sklearn.metrics import mutual_info_score as mi

import argparse
import pickle as pkl

from math import log

def to_list_of_lists(assignments):
	all_classes = set(assignments)
	num_classes = len(all_classes)
	l = []
	for class_ in all_classes:
		idx, = np.where(assignments == class_)
		l.append(idx)
	return l

def variation_of_information(X, Y):

	# Variation of information (VI)
	#
	# Meila, M. (2007). Comparing clusterings-an information
	#   based distance. Journal of Multivariate Analysis, 98,
	#   873-895. doi:10.1016/j.jmva.2006.11.013
	#
	# https://en.wikipedia.org/wiki/Variation_of_information
	# https://gist.github.com/jwcarr/626cbc80e0006b526688

	n = float(sum([len(x) for x in X]))
	sigma = 0.0
	for x in X:
		p = len(x) / n
	for y in Y:
		q = len(y) / n
		r = len(set(x) & set(y)) / n
		if r > 0.0:
			sigma += r * (log(r / p, 2) + log(r / q, 2))
	return abs(sigma)


# def vi(true_labels, predicted_labels):

# 	def convert_to_1d_labels(labels):
# 		ecount = count()
# 		enames = ddict(ecount.__next__)
# 		return [enames[i] for i in labels]

# 	true_labels = convert_to_1d_labels(true_labels)
# 	predicted_labels = convert_to_1d_labels(predicted_labels)

# 	true_probs = np.array(list(Counter(true_labels).values()))
# 	predicted_probs = np.array(list(Counter(predicted_labels).values()))

# 	true_probs = true_probs / true_probs.sum()
# 	predicted_probs = predicted_probs / predicted_probs.sum()

# 	return sp.stats.entropy(true_probs) + sp.stats.entropy(predicted_probs) - 2 * mi(true_labels, predicted_labels)

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Evaluate core-periphery detections algorithms")

	parser.add_argument("--edgelist_dir", dest="edgelist_directory", type=str, default="datasets/",
		help="The directory containing edgelist files (default is 'datasets/').")
	parser.add_argument("--prediction_dir", dest="prediction_directory", type=str, default="predictions/",
		help="The directory to load predictions (default is 'predictions/').")
	parser.add_argument("--results_dir", dest="results_directory", type=str, default="results/",
		help="The directory to output results (default is 'results/').")

	# parser.add_argument("--exp", dest="exp", type=str, default="one_core",
	# 	help="The experiment type (default is 'one_core').")
	# parser.add_argument("--algorithm", dest="algorithm", type=str, default="BE",
	# 	help="The algorithm (default is 'BE').")
	# parser.add_argument("--theta1", dest="theta1", type=float, default=0.1,
	# 	help="Value of theta1 (default is 0.1).")
	# parser.add_argument("--theta2", dest="theta2", type=float, default=0.05,
	# 	help="Value of theta2 (default is 0.05).")

	args = parser.parse_args()
	return args

def main():
	
	args = parse_args()

	edgelist_dir = args.edgelist_directory
	predictions_dir = args.prediction_directory

	num_seeds = 255

	# exp = args.exp
	algorithms = ["BE", "divisive", "km_ER", "km_config"]

	results = pd.DataFrame(0., index=algorithms, columns=["assigned_component", "SCC"])

	# for theta1 in np.arange(0.10, 1.0, 0.05):
	# 	for theta2 in np.arange(0.05, theta1, 0.05):
	for algorithm in algorithms:

		vis_assigned = np.zeros(num_seeds)
		vis_SCC = np.zeros(num_seeds)

		for seed in range(num_seeds):

			true_labels_directory = os.path.join(edgelist_dir, "synthetic_bow_tie",)
			true_labels_filename = "seed={:03d}".format(seed)
			node_label_filename = os.path.join(true_labels_directory, true_labels_filename + ".csv")

			true_labels = pd.read_csv(node_label_filename, index_col=0)

			true_labels_assigned_core = [true_labels.at[n, "assigned_component"] == 0 for n in sorted(true_labels.index)]
			true_labels_scc = [true_labels.at[n, "SCC"] >= 0 for n in sorted(true_labels.index)]

			true_labels_assigned_core = to_list_of_lists(true_labels_assigned_core)
			true_labels_scc = to_list_of_lists(true_labels_scc)


			predicted_labels_directory = os.path.join(predictions_dir, "synthetic_bow_tie", algorithm)
			predicted_label_filename = os.path.join(predicted_labels_directory,
				"seed={:03d}.csv".format(seed) )

			predicted_labels = pd.read_csv(predicted_label_filename, index_col=0)
			predicted_labels = [predicted_labels.at[n, "is_core"] for n in sorted(predicted_labels.index)]
			predicted_labels = to_list_of_lists(predicted_labels)

			# vis_assigned[seed] = vi(true_labels_assigned_core, predicted_labels)
			# vis_SCC[seed] = vi(true_labels_scc, predicted_labels)


			vis_assigned[seed] = variation_of_information(true_labels_assigned_core, predicted_labels)
			vis_SCC[seed] = variation_of_information(true_labels_scc, predicted_labels)


		results.at[algorithm, "assigned_component"] = vis_assigned.mean()
		results.at[algorithm, "SCC"] = vis_SCC.mean()

	print (results.to_string())

	results_directory = args.results_directory
	results_directory = os.path.join(results_directory, "synthetic_bow_tie", )
	if not os.path.exists(results_directory):
		os.makedirs(results_directory, exist_ok=True)
		print ("Making: {}".format(results_directory))
	results_filename = os.path.join(results_directory, "vi_scores.csv")
	results.to_csv(results_filename, sep=",")


if __name__ == "__main__":
	main()