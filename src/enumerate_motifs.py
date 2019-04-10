import numpy as np
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
	print ("found {} paths".format(len(paths)))
	return paths

def parse_args():
	parser = argparse.ArgumentParser(description="Enumerate all motifs in a given network")

	parser.add_argument("--embedding", dest="embedding_filename",  
		help="path of embedding to load.")
	parser.add_argument("--edgelist", dest="edgelist", type=str,
		help="The edgelist of the graph.")
	parser.add_argument("--features", dest="features", type=str, default="none",
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str,
		help="path to labels")
	
	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

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

	poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
	ranks = 2 * np.arctanh(np.linalg.norm(poincare_embedding, axis=-1))

	print ("core ranks:\nmin={}\nmean={}\nmax={}\n".format(ranks[labels==0].min(),
		ranks[labels==0].mean(), ranks[labels==0].max()))
	print ("periphery-in ranks:\nmin={}\nmean={}\nmax={}\n".format(ranks[labels==1].min(),
		ranks[labels==1].mean(), ranks[labels==1].max()))
	print ("periphery-out ranks:\nmin={}\nmean={}\nmax={}\n".format(ranks[labels==2].min(),
		ranks[labels==2].mean(), ranks[labels==2].max()))
	print ()

	motif_size = 3

	for motif in motifs.values():
		assert len(motif) == motif_size

	path_labels = enumerate_motif_occurences(graph,  )
	found_motifs = {name: [] for name in motifs.keys()}
	for path, name in path_labels:
		if name is not None:
			found_motifs[name].append(path)

	for k, v in found_motifs.items():
		print ("Found {} instances of motif {}".format(len(v), k))
		np.savetxt(fname="test_{}.csv".format(k), X=np.array(v), delimiter=",", fmt="%d")

	print ()
	print ("Testing statistical significances of motifs")

	n = len(graph)
	p = len(graph.edges()) / n ** 2
	num_erdos_renyi_graphs = args.num_erdos_renyi_graphs
	print ("Generating {} ER graphs with n={} and p={}".format(num_erdos_renyi_graphs, n, p))
	erdos_renyi_graphs = generate_erdos_renyi(n, p, num_graphs=num_erdos_renyi_graphs)
	erdos_renyi_num_occurences = {name: np.zeros(num_erdos_renyi_graphs) for name in motifs.keys()}
	for i, erdos_renyi_graph in enumerate(erdos_renyi_graphs):
		print ("number of nodes: {}, number of edges: {}".format(len(erdos_renyi_graph), len(erdos_renyi_graph.edges())))
		# erdos_renyi_paths = enumerate_paths(erdos_renyi_graph, motif_size)
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
	print ("Enumerating motifs in order of significance")
	node_counts = {name: np.zeros(n, dtype=np.int) for name in motifs}
	# motif_mean_ranks = {name: np.inf for name in motifs}
	for name in sorted(p_values, key=p_values.get):
		if p_values[name] > 0.05:
			continue
		print ("motif: {}\np-value: {}\n".format(name, p_values[name]))
		for path in found_motifs[name]:
			node_counts[name][list(path)] += 1

		counts = node_counts[name]
		print ("num occurences: {}".format(counts.sum() / 3))
		print ("num occurences in core: {}".format(counts[labels==0].sum() / 3))
		print ("num occurences in periphery-in: {}".format(counts[labels==1].sum() / 3))
		print ("num occurences in periphery-out: {}".format(counts[labels==2].sum() / 3))

		motif_ranks = ranks[np.array(found_motifs[name]).flatten()]
		print ("min rank: {}\nmean rank: {}\nmax rank: {}\n".format(motif_ranks.min(), motif_ranks.mean(), motif_ranks.max()))

if __name__ == "__main__":
	main()