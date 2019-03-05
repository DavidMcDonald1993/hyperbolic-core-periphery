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

def vi(true_labels, predicted_labels):
	def convert_to_1d_labels(labels):
		ecount = count()
		enames = ddict(ecount.__next__)
		return [enames[i] for i in labels]

	true_labels = convert_to_1d_labels(true_labels)
	predicted_labels = convert_to_1d_labels(predicted_labels)

	true_probs = np.array(list(Counter(true_labels).values()))
	predicted_probs = np.array(list(Counter(predicted_labels).values()))

	true_probs = true_probs / true_probs.sum()
	predicted_probs = predicted_probs / predicted_probs.sum()

	return sp.stats.entropy(true_probs) + sp.stats.entropy(predicted_probs) - 2 * mi(true_labels, predicted_labels)

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Evaluate core-periphery detections algorithms")

	parser.add_argument("--edgelist_dir", dest="edgelist_directory", type=str, default="edgelists/",
		help="The directory containing edgelist files (default is 'edgelists/').")
	parser.add_argument("--prediction_dir", dest="prediction_directory", type=str, default="predictions/",
		help="The directory to load predictions (default is 'predictions/').")
	parser.add_argument("--results_dir", dest="results_directory", type=str, default="results/",
		help="The directory to output results (default is 'results/').")

	parser.add_argument("--exp", dest="exp", type=str, default="one_core",
		help="The experiment type (default is 'one_core').")
	parser.add_argument("--algorithm", dest="algorithm", type=str, default="BE",
		help="The algorithm (default is 'BE').")
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

	num_seeds = 30

	exp = args.exp
	algorithm = args.algorithm

	columns = ["{:.02f}".format(i) for i in np.arange(0.05, 1, 0.05)]
	rows = ["{:.02f}".format(i) for i in np.arange(0.05, 1, 0.05)]

	results = pd.DataFrame(np.nan, columns=columns, index=rows)

	for theta1 in np.arange(0.10, 1.0, 0.05):
		for theta2 in np.arange(0.05, theta1, 0.05):

			vis = np.zeros(num_seeds)

			for seed in range(num_seeds):

				true_labels_directory = os.path.join(edgelist_dir, "synthetic_core_periphery", exp)
				true_labels_filename = "theta1={:.02f}-theta2={:.02f}-seed={:02d}".format(theta1, theta2, seed)
				node_label_filename = os.path.join(true_labels_directory, true_labels_filename + ".pkl")

				with open(node_label_filename, "rb") as f:
					true_labels = pkl.load(f)

				true_labels = [true_labels[n] for n in range(len(true_labels))]
				# print (true_labels)
				# raise SystemError

				predicted_labels_directory = os.path.join(predictions_dir, "synthetic_core_periphery", exp, algorithm)
				predicted_label_filename = os.path.join(predicted_labels_directory,
					"theta1={:.02f}-theta2={:.02f}-seed={:02d}.pkl".format(theta1, theta2, seed) )

				with open(predicted_label_filename, "rb") as f:
					predicted_labels = pkl.load(f)

				predicted_labels = [predicted_labels[n] for n in range(len(predicted_labels))]


				vis[seed] = vi(true_labels, predicted_labels)

			results.at["{:.02f}".format(theta2), "{:.02f}".format(theta1)] = vis.mean()

	# print (results.shape)
	results_directory = args.results_directory
	results_directory = os.path.join(results_directory, "synthetic_core_periphery", exp, algorithm)
	if not os.path.exists(results_directory):
		os.makedirs(results_directory, exist_ok=True)
		print ("Making: {}".format(results_directory))
	results_filename = os.path.join(results_directory, "vi_scores.csv")
	results.to_csv(results_filename, sep=",")


if __name__ == "__main__":
	main()