import sys
if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")
import os 


import numpy as np
import networkx as nx
import pandas as pd

import cpalgorithm as cp

import pickle as pkl

import argparse

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Run core-periphery detections algorithms")

	parser.add_argument("--edgelist_directory", dest="edgelist_directory", type=str, default="datasets",
		help="The path to the predictions_directory containing edgelist files (default is 'datasets/').")
	parser.add_argument("--prediction_dir", dest="prediction_directory", type=str, default="predictions/",
		help="The predictions_directory to output predictions (default is 'predictions/').")



	# parser.add_argument("--seed", dest="seed", type=int, default=0,
	# 	help="Random seed (default is 0).")

	# parser.add_argument("--algorithm", dest="algorithm", type=str, default="BE",
		# help="The algorithm (default is 'BE').")

	args = parser.parse_args()
	return args


def main():

	args = parse_args()

	edgelist_dir = args.edgelist_directory
	predictions_dir = args.prediction_directory

	num_seeds = 255

	# seed = args.seed
	# exp = args.exp
	# theta1 = args.theta1
	# theta2 = args.theta2
	# algorithm = args.algorithm

	# exps = ["one_core", "two_core", "two_core_with_residual"]
	algorithms = ["BE", "divisive", "km_ER", "km_config"]

	# assert exp in exps
	# assert algorithm in algorithms
	# assert seed < num_seeds

	# for theta1 in np.arange(0.10, 1.0, 0.05):
	# 	for theta2 in np.arange(0.05, theta1, 0.05):

	for algorithm in algorithms:

		for seed in range(num_seeds):

			edgelist_filename = os.path.join(edgelist_dir, "synthetic_bow_tie", 
				"seed={:03d}.tsv".format(seed) )

			predictions_directory = os.path.join(predictions_dir, "synthetic_bow_tie", algorithm)

			if not os.path.exists(predictions_directory):
				print ("Making predictions_directory: {}".format(predictions_directory))
				os.makedirs(predictions_directory, exist_ok=True)
			
			output_filename = os.path.join(predictions_directory,
				"seed={:03d}.csv".format(seed) )

			# if os.path.exists(output_filename):
			# 	print("{} already exists, continuing".format(output_filename))
			# 	continue

			graph = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=str)


			if algorithm == "BE":
				_algorithm = cp.BE()
			elif algorithm == "divisive":
				_algorithm = cp.Divisive()
			elif algorithm == "km_ER":
				_algorithm = cp.KM_ER()
			elif algorithm == "km_config":
				_algorithm = cp.KM_config()
			_algorithm.detect(graph)

			c = _algorithm.get_pair_id()
			x = _algorithm.get_coreness()

			predicted_labels = {n : (c[str(n)], int(x[str(n)])) for n in sorted(graph.nodes())}

			predicted_labels = pd.DataFrame.from_dict(predicted_labels, orient="index")
			predicted_labels.columns = ["core_number", "is_core"]
			predicted_labels.to_csv(output_filename)


			# sig_c, sig_x, significant, p_values = cp.qstest(c, x, graph, _algorithm)

			# print (sig_c, sig_x, significant, p_values)
			# raise SystemExit


			print ("Completed {}".format(output_filename))

if __name__ == "__main__":
	main()