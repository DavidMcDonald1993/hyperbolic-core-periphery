import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score

import argparse

from utils import load_data, hyperboloid_to_poincare_ball, hyperbolic_distance

def parse_args():
	parser = argparse.ArgumentParser(description="Identfy three componenets of bow-tie structure from hyperboloid embedding")

	parser.add_argument("--embedding", dest="embedding_filename",  
		help="path of embedding to load.")
	parser.add_argument("--edgelist", dest="edgelist", type=str,
		help="The edgelist of the graph.")
	parser.add_argument("--features", dest="features", type=str, default="none",
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str,
		help="path to labels")
	
	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')


	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	assert args.directed

	graph, features, labels, hyperboloid_embedding = load_data(args)
	print ("number of nodes: {}, number of edges: {}".format(len(graph), len(graph.edges())))

	poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
	ranks = 2 * np.arctanh(np.linalg.norm(poincare_embedding, axis=-1))

	cp_clustering = KMeans(n_clusters=2).fit(ranks[:,None])

	print (normalized_mutual_info_score(labels==0, cp_clustering.labels_))

	periphery_embedding = hyperboloid_embedding[cp_clustering.labels_ == cp_clustering.labels_[ranks.argmax()]]
	
	dists = hyperbolic_distance(periphery_embedding, periphery_embedding)
	in_out_clustering = AgglomerativeClustering(n_clusters=2, 
		affinity="precomputed",
		linkage="average").fit(dists)

	print (normalized_mutual_info_score(labels[labels>0], in_out_clustering.labels_))


if __name__ == "__main__":
	main()