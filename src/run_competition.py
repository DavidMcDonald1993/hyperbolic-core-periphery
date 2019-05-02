import sys
if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")
import os 


import numpy as np
import networkx as nx

import cpalgorithm as cp

import pickle as pkl

import argparse

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Run core-periphery detections algorithms")

	parser.add_argument("--edgelist_dir", dest="edgelist_directory", type=str, default="edgelists/",
		help="The directory containing edgelist files (default is 'edgelists/').")
	parser.add_argument("--prediction_dir", dest="prediction_directory", type=str, default="predictions/",
		help="The directory to output predictions (default is 'predictions/').")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")
	parser.add_argument("--exp", dest="exp", type=str, default="one_core",
		help="The experiment type (default is 'one_core').")
	# parser.add_argument("--theta1", dest="theta1", type=float, default=0.1,
	# 	help="Value of theta1 (default is 0.1).")
	# parser.add_argument("--theta2", dest="theta2", type=float, default=0.05,
	# 	help="Value of theta2 (default is 0.05).")
	parser.add_argument("--algorithm", dest="algorithm", type=str, default="BE",
		help="The algorithm (default is 'BE').")



	args = parser.parse_args()
	return args


def main():

	args = parse_args()

	edgelist_dir = args.edgelist_directory
	predictions_dir = args.prediction_directory

	num_nodes = 400
	num_seeds = 30

	seed = args.seed
	exp = args.exp
	# theta1 = args.theta1
	# theta2 = args.theta2
	algorithm = args.algorithm

	exps = ["one_core", "two_core", "two_core_with_residual"]
	algorithms = ["BE", "divisive", "km_ER", "km_config"]

	assert exp in exps
	assert algorithm in algorithms
	assert seed < num_seeds
	# assert theta2 < theta1
	# if theta2 >= theta1:
	# 	return

	# for exp in exps:

	for theta1 in np.arange(0.10, 1.0, 0.05):
		for theta2 in np.arange(0.05, theta1, 0.05):

	# 			for algorithm in algorithms:

	# 				for seed in range(num_seeds):

			edgelist_filename = os.path.join(edgelist_dir, "synthetic_core_periphery", exp,
				"theta1={:.02f}-theta2={:.02f}-seed={:02d}.edgelist".format(theta1, theta2, seed) )

			directory = os.path.join(predictions_dir, "synthetic_core_periphery", exp, algorithm)

			if not os.path.exists(directory):
				print ("Making directory: {}".format(directory))
				os.makedirs(directory, exist_ok=True)
			
			label_filename = os.path.join(directory,
				"theta1={:.02f}-theta2={:.02f}-seed={:02d}.pkl".format(theta1, theta2, seed) )

			if os.path.exists(label_filename):
				print("{} already exists, continuing".format(label_filename))
				continue

			g = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=str)

			# with open("edgelists/synthetic/one_core/theta1=0.10-theta2=0.05-seed=00.pkl", "rb") as f:
			# 	node_labels = pkl.load(f)

			if algorithm == "BE":
				_algorithm = cp.BE()
			elif algorithm == "divisive":
				_algorithm = cp.Divisive()
			elif algorithm == "km_ER":
				_algorithm = cp.KM_ER()
			elif algorithm == "km_config":
				_algorithm = cp.KM_config()
			_algorithm.detect(g)

			c = _algorithm.get_pair_id()
			x = _algorithm.get_coreness()

			predicted_labels = [(c[str(n)], x[str(n)]) for n in range(len(g))]

			with open(label_filename, "wb") as f:
				pkl.dump(predicted_labels, f, pkl.HIGHEST_PROTOCOL)

			print ("Completed {}".format(label_filename))
	# sig_c, sig_x, significant, p_values = cp.qstest(c, x, g, algorithm)

if __name__ == "__main__":
	main()