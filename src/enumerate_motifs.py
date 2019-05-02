import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import argparse
import functools

from multiprocessing import Pool

from networkx.algorithms.isomorphism import DiGraphMatcher

from scipy.stats import norm

from sklearn.cluster import KMeans, AgglomerativeClustering

from utils import load_data, hyperboloid_to_poincare_ball, hyperbolic_distance
from visualise import draw_graph
from clustering_hyperboloid import compute_centroid

CUTOFF = 0.05

motifs = {
	"m1": nx.DiGraph( ((0, 1), (1, 2), (2, 0)) ),
	"m2": nx.DiGraph( ((0, 1), (1, 2), (2, 0), (0, 2)) ),
	"m3": nx.DiGraph( ((0, 1), (1, 2), (2, 0), (0, 2), (2, 1)) ),
	"m4": nx.DiGraph( ((0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)) ),
	"m5": nx.DiGraph( ((0, 1), (1, 2), (0, 2)) ),
	"m6": nx.DiGraph( ((0, 1), (1, 2), (0, 2), (2, 1)) ),
	"m7": nx.DiGraph( ((1, 0), (1, 2), (2, 0), (2, 1)) ),
	"m8": nx.DiGraph( ((0, 1), (0, 2),) ),
	"m9": nx.DiGraph( ((0, 1), (2, 0),) ),
	"m10": nx.DiGraph( ((1, 0), (2, 0),) ),
	"m11": nx.DiGraph( ((1, 0), (2, 0), (0, 1)) ),
	"m12": nx.DiGraph( ((1, 0), (0, 2), (0, 1)) ),
	"m13": nx.DiGraph( ((1, 0), (0, 2), (0, 1), (2, 0)) ),
}


def generate_erdos_renyi(n, p, num_graphs=100):
	return (nx.erdos_renyi_graph(n, p, seed=seed, directed=True) for seed in range(num_graphs))

def find_paths(u, n, g):
	if n==0:
		return [[u]]
	paths = [[u]+path 
	for neighbor in g.neighbors(u) 
	for path in find_paths(neighbor, n-1, g) 
	if u not in path ]
	return paths

def check_subgraph(nodes, graph, ):
	s = graph.subgraph(nodes)
	for name, motif in motifs.items():
		if DiGraphMatcher(s, motif).is_isomorphic():
			return nodes, name
	return nodes, None

def build_motif_adj(graph, found_motifs):

	n = len(graph)

	motif_adj = np.zeros((n, n))

	for found_motif in found_motifs:
		for u in found_motif:
			for v in found_motif:
				if u == v:
					continue
				motif_adj[u, v] += 1
	return motif_adj


def enumerate_motif_occurences(graph, ):
	paths = enumerate_paths(graph, len(motifs["m1"]))

	with Pool(processes=None) as p:
		is_motif = p.map(functools.partial(check_subgraph, 
			graph=graph, ),
			paths,
			)
	return is_motif

def enumerate_paths(graph, n):
	with Pool(processes=None) as p:
		paths = p.map(functools.partial(find_paths, n=n-1, g=graph.to_undirected()),
			graph.nodes())
	paths = [tuple(sorted(path)) for nodes in paths for path in nodes ]
	paths = list(set(paths))
	print ("enumerated all {}-node paths in graph".format(n))
	print ("found {} potential motifs".format(len(paths)))
	return paths

def parse_args():
	parser = argparse.ArgumentParser(description="Enumerate all motifs in a given network")

	parser.add_argument("--embedding", dest="embedding_filename",  
		help="path of embedding to load.")
	parser.add_argument("--edgelist", dest="edgelist", type=str,
		help="The edgelist of the graph.")
	parser.add_argument("--features", dest="features", type=str,
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str,
		help="path to labels")

	parser.add_argument("--motifs", dest="motifs", type=str,
		help="path to save found motifs")
	
	parser.add_argument('--directed', action="store_true", help='flag for directed graph')

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	parser.add_argument("--num-er", dest="num_erdos_renyi_graphs", type=int, default=100,
		help="Number of ER graphs to generate for testing statistical significance of motif occurence (default is 100).")

	args = parser.parse_args()
	return args

def main():

	print ()

	args = parse_args()
	assert args.directed

	graph, features, labels, hyperboloid_embedding = load_data(args)
	print ("Loaded graph:\nnumber of nodes: {}, number of edges: {}\n".format(len(graph), len(graph.edges())))

	if not os.path.exists(args.motifs):
		os.makedirs(args.motifs, exist_ok=True)
	motif_statistics_filename= os.path.join(args.motifs, "motif_statistics.csv")
	path_labels_filename = os.path.join(args.motifs, "path_labels.csv")

	poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
	ranks = 2 * np.arctanh(np.linalg.norm(poincare_embedding, axis=-1))

	# print ("core ranks:\nmin={}\nmean={}\nmax={}\n".format(ranks[labels==0].min(),
	# 	ranks[labels==0].mean(), ranks[labels==0].max()))
	# print ("periphery-in ranks:\nmin={}\nmean={}\nmax={}\n".format(ranks[labels==1].min(),
	# 	ranks[labels==1].mean(), ranks[labels==1].max()))
	# print ("periphery-out ranks:\nmin={}\nmean={}\nmax={}\n".format(ranks[labels==2].min(),
	# 	ranks[labels==2].mean(), ranks[labels==2].max()))
	# print ()

	motif_size = 3

	for motif in motifs.values():
		assert len(motif) == motif_size

	path_labels = enumerate_motif_occurences(graph,  )

	found_motifs = {name: [] for name in motifs.keys()}
	with open(path_labels_filename, "w") as f:
		for path, name in path_labels:
			if name is not None:
				found_motifs[name].append(path)
				f.write("{}\t{}\t{}\t{}\n".format(*path, name))

	print ()
	print ("Testing statistical significances of motifs\n")

	n = len(graph)
	p = len(graph.edges()) / n ** 2
	if not args.directed:
		p = 2 * p
	num_erdos_renyi_graphs = args.num_erdos_renyi_graphs
	print ("Generating {} ER graphs with n={} and p={}".format(num_erdos_renyi_graphs, n, p))
	erdos_renyi_graphs = generate_erdos_renyi(n, p, num_graphs=num_erdos_renyi_graphs)
	erdos_renyi_num_occurences = {name: np.zeros(num_erdos_renyi_graphs) for name in motifs.keys()}
	for i, erdos_renyi_graph in enumerate(erdos_renyi_graphs):
		print ("number of nodes: {}, number of edges: {}".format(len(erdos_renyi_graph), len(erdos_renyi_graph.edges())))
		erdos_renyi_path_labels = enumerate_motif_occurences(erdos_renyi_graph,  )
		for _, name in erdos_renyi_path_labels:
			if name is not None:
				erdos_renyi_num_occurences[name][i] += 1
		print ("Completed random graph {}\n".format(i))

	print ()
	p_values = {name: 1. for name in motifs}
	for name, counts in erdos_renyi_num_occurences.items():
		mean = counts.mean()
		std = counts.std()
		if std == 0:
			print ("motif: {} std is 0 -- passing".format(name))
			continue
		dist = norm(mean, std)
		num_occurences_in_graph = len(found_motifs[name])
		if num_occurences_in_graph < mean:
			print ("motif {} is under-represented\n".format(name))
			continue
		p_value = dist.pdf(num_occurences_in_graph)
		print ("motif: {}\nmean: {}\nstd: {}\nnum occurences: {}\np-value: {}\n".format(name, mean, std, num_occurences_in_graph, p_value))
		p_values[name] = p_value

	print ()
	print ("Enumerating motifs in order of significance\n")

	motif_statistics_df = pd.DataFrame(columns=["p-value", 
		# "num_occurences_core", "num_occurences_periphery_in", "num_occurences_periphery_out",
		"min_rank", "mean_rank", "max_rank"])
	node_counts = {name: np.zeros(n, dtype=np.int) for name in motifs}
	for name in sorted(p_values, key=p_values.get):
		if p_values[name] > CUTOFF:
			continue
		p_value = p_values[name]
		print ("motif: {}\np-value: {}\n".format(name, p_value))
		for path in found_motifs[name]:
			node_counts[name][list(path)] += 1

		counts = node_counts[name]
		
		# num_occurences_core = counts[labels==0].sum() / 3
		# num_occurences_periphery_in = counts[labels==1].sum() / 3
		# num_occurences_periphery_out = counts[labels==2].sum() / 3

		print ("num occurences: {}".format(counts.sum() / 3))
		# print ("num occurences in core: {}".format(num_occurences_core))
		# print ("num occurences in periphery-in: {}".format(num_occurences_periphery_in))
		# print ("num occurences in periphery-out: {}".format(num_occurences_periphery_out))

		# motif_ranks = ranks[np.array(found_motifs[name]).flatten()]
		# print ("min rank: {}\nmean rank: {}\nmax rank: {}\n".format(motif_ranks.min(), motif_ranks.mean(), motif_ranks.max()))
		centroids = np.concatenate([compute_centroid(hyperboloid_embedding[list(idx)]) 
			for idx in found_motifs[name]])
		print ("computed centroids of all instances of {}".format(name))

		centroid_ranks = 2 * np.arctanh(np.linalg.norm(hyperboloid_to_poincare_ball(centroids), axis=-1))

		motif_statistics_df.loc[name] = {
			"p-value": p_value, 
			# "num_occurences_core": num_occurences_core,
			# "num_occurences_periphery_in": num_occurences_periphery_in,
			# "num_occurences_periphery_out": num_occurences_periphery_out,
			"min_rank": centroid_ranks.min(),
			"mean_rank": centroid_ranks.mean(),
			"max_rank": centroid_ranks.max()
		}

		# build motif adjacency matrix for all statistically significant motifs
		print ("building motif adjacency matrix for motif {}".format(name))
		motif_adj = build_motif_adj(graph, found_motifs[name])

		motif_graph = nx.from_numpy_matrix(motif_adj)
		motif_graph.add_edges_from(((u, u, {"weight": 0.}) for u in graph.nodes() ))
		nx.write_edgelist(motif_graph, 
			os.path.join("edgelists", "ecoli", "{}_motif_graph.tsv".format(name)), 
			delimiter="\t", data=["weight"])

	print("writing statistics to {}".format(motif_statistics_filename))
	motif_statistics_df.to_csv(motif_statistics_filename, sep=",")

if __name__ == "__main__":
	main()