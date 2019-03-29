import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import argparse
import functools

from multiprocessing import Pool

from networkx.algorithms.isomorphism import DiGraphMatcher

from utils import load_data

def find_paths(u, n, g):
	if n==0:
		return [[u]]
	paths = [[u]+path for neighbor in g.neighbors(u) for path in find_paths(neighbor, n-1, g) if u not in path]
	return paths

def check_subgraph(nodes, g, motif):
	s = g.subgraph(nodes)
	return nx.DiGraph(s) if DiGraphMatcher(s, motif).is_isomorphic() else None

def build_motif_adj(g, found_motifs):

	n = len(g)

	motif_adj = np.zeros((n, n))

	for found_motif in found_motifs:

		for u in found_motif:
			for v in found_motif:
				if u == v:
					continue
				motif_adj[u, v] += 1
	return motif_adj


def enumerate_motif_occurences(g, motif):
	assert isinstance(g, nx.DiGraph)
	assert isinstance(motif, nx.DiGraph)
	# assert len(motif) == 3

	n = len(motif)

	with Pool(processes=None) as p:
		paths = p.map(functools.partial(find_paths, n=n-1, g=g.to_undirected()),
			g.nodes())
	paths = np.concatenate(paths, axis=0)
	print ("enumerated all {}-node paths in g".format(n))
	print ("found {} paths".format(paths.shape[0]))
	with Pool(processes=None) as p:
		is_motif = p.map(functools.partial(check_subgraph, 
			g=g, 
			motif=motif), 
			paths,
			)
	# p.close()
	# p.join()

	return filter(lambda x: x is not None, is_motif)

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

	args = parser.parse_args()
	return args

def main():

	args = parse_args()
	assert args.directed

	graph, features, labels, hyperboloid_embedding = load_data(args)
	# graph = graph.subgraph([n for n in graph if labels[n] == 0])
	# graph = nx.convert_node_labels_to_integers(graph)
	print ("number of nodes: {}, number of edges: {}".format(len(graph), len(graph.edges())))


	motif = nx.DiGraph([(0, 2), (1, 2), ])

	found_motifs = enumerate_motif_occurences(graph, motif)

	print ("Found {} instances of motif".format(len(list(found_motifs))))

	motif_adj = build_motif_adj(graph, found_motifs)

	plt.imshow(motif_adj)
	plt.show()

	# print (len(list (found_motifs)))
	# for _motif in found_motifs:
	# 	pos = nx.spring_layout(_motif)
	# 	nx.draw_networkx_nodes(_motif, pos=pos,)
	# 	nx.draw_networkx_edges(_motif, pos=pos)
	# 	nx.draw_networkx_labels(_motif, pos=pos)
	# 	plt.show()

if __name__ == "__main__":
	main()