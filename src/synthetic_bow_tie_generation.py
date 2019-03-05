import sys
if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")

import numpy as np
import networkx as nx

# import cpalgorithm as cp
# import json
import os
import pickle as pkl

import argparse

def build_bow_tie(num_nodes, num_edges, seed=0):
    
    def to_directed(g, direction=lambda u, v: v >= u):
        if nx.is_directed(g):
            return g
        degrees = dict(nx.degree(g))
        return nx.DiGraph([(u, v) if direction(degrees[u], degrees[v]) else (v, u) for u, v in g.edges()])

    num_core = int(np.ceil(num_nodes/10.))
    num_nodes += num_core # account for overlap
    num_periphery_in = int(np.ceil(num_nodes/2))
    num_periphery_out = num_nodes - num_periphery_in

    periphery_in = to_directed(nx.barabasi_albert_graph(num_periphery_in, 
        m=num_edges, seed=2*seed))

    periphery_in_degrees = dict(periphery_in.degree())
    periphery_in_nodes_sorted = sorted(periphery_in_degrees, key=periphery_in_degrees.get, reverse=True)
    periphery_in_core_nodes = periphery_in_nodes_sorted[:num_core]
    periphery_in_periphery_nodes = periphery_in_nodes_sorted[num_core:]

    # relabel core
    periphery_in = nx.relabel_nodes(periphery_in, {n: "core_{}".format(i) for i, n in enumerate(periphery_in_core_nodes)})
    # relabel periphery
    periphery_in = nx.relabel_nodes(periphery_in, {n: "periphery_in_{}".format(i) for i, n in enumerate(periphery_in_periphery_nodes)})

    periphery_out = to_directed(nx.barabasi_albert_graph(num_periphery_out, 
        m=num_edges, seed=2*seed+1), direction=lambda u, v: v <= u)

    periphery_out_degrees = dict(periphery_out.degree())
    periphery_out_nodes_sorted = sorted(periphery_out_degrees, key=periphery_out_degrees.get, reverse=True)
    periphery_out_core_nodes = periphery_out_nodes_sorted[:num_core]
    periphery_out_periphery_nodes = periphery_out_nodes_sorted[num_core:]

    # relabel core
    periphery_out = nx.relabel_nodes(periphery_out, {n: "core_{}".format(i) for i, n in enumerate(periphery_out_core_nodes)})
    # relabel periphery
    periphery_out = nx.relabel_nodes(periphery_out, {n: "periphery_out_{}".format(i) for i, n in enumerate(periphery_out_periphery_nodes)})

    graph = periphery_in.copy()
    graph.add_edges_from(periphery_out.edges())
    nx.set_edge_attributes(graph, values=1., name="weight")

    mapping = {n : i for i, n in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)

    node_labels = {v: k for k, v in mapping.items()}

    return node_labels, graph

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Run core-periphery detections algorithms")

	parser.add_argument("--edgelist_dir", dest="edgelist_directory", type=str, default="edgelists/",
		help="The directory containing edgelist files (default is 'edgelists/').")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")
	parser.add_argument("-m", "--num-edges", dest="num_edges", type=int, default=1,
		help="Number of new edges per new node (default is 1).")

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	edgelist_directory = args.edgelist_directory

	num_nodes = 400
	num_seeds = 30

	seed = args.seed
	num_edges = args.num_edges

	directory = os.path.join(edgelist_directory, "synthetic_bow_tie")
	if not os.path.exists(directory):
		print ("Making directory: {}".format(directory))
		os.makedirs(directory, exist_ok=True)

	filename = "num_edges={:02d}-seed={:02d}".format(num_edges, seed)
	edgelist_filename = os.path.join(directory, filename + ".edgelist")
	node_label_filename = os.path.join(directory, filename + ".pkl")

	if os.path.exists(edgelist_filename):
		print ("{} already exists".format(edgelist_filename))
		return

	node_labels, graph = build_bow_tie(num_nodes, 
		num_edges=num_edges, 
		seed=seed)

	nx.write_edgelist(graph, edgelist_filename, delimiter="\t")
	with open(node_label_filename, "wb") as f:
		pkl.dump(node_labels, f, pkl.HIGHEST_PROTOCOL)

	print ("Completed {}".format(filename))

if __name__ == "__main__":
	main()
