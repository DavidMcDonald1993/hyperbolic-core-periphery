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

	parser.add_argument("--edgelist_directory", dest="edgelist_directory", type=str, default="edgelists",
		help="The path to the predictions_directory containing edgelist files (default is 'edgelists/').")
	parser.add_argument("--output_dir", dest="output_directory", type=str, default="condensed_bow_ties/",
		help="The directory to output condensed graphs (default is 'condensed_bow_ties/').")

	args = parser.parse_args()
	return args


def main():

	args = parse_args()

	edgelist_dir = args.edgelist_directory
	output_dir = args.output_directory

	num_seeds = 255

	for seed in range(num_seeds):

		edgelist_filename = os.path.join(edgelist_dir, "synthetic_bow_tie", 
			"seed={:03d}.tsv".format(seed) )

		graph_output_filename = os.path.join(output_dir,
			"seed={:03d}.tsv".format(seed) )
		labels_filename = os.path.join(output_dir,
			"seed={:03d}.csv".format(seed) )

		graph = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=str, create_using=nx.DiGraph())

		graph = 



		print ("Completed {}".format(graph_output_filename))

if __name__ == "__main__":
	main()