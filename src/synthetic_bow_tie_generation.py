import sys
if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")

import numpy as np
import networkx as nx
import pandas as pd

import os
import pickle as pkl

import argparse

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

def build_bow_tie(num_nodes, core_prob, connection_probs, 
								 self_loops=False, directed=True, seed=0):

	def change_edge_direction(graph, direction=lambda u, v: v >= u):
		if nx.is_directed(graph):
			return graph
		degrees = dict(nx.degree(graph))
		return nx.DiGraph([(u, v) if direction(degrees[u], degrees[v]) else (v, u) for u, v in graph.edges()])

	np.random.seed(seed)

	assert connection_probs.shape[0] == 3
	assert connection_probs.shape[1] == 3

	periphery_prob = 1 - core_prob

	num_in = int(np.ceil(num_nodes * periphery_prob / 2.))
	num_core = int(np.ceil(num_nodes * core_prob))
	num_out = num_nodes - (num_in + num_core)

	node_labels = {n : (1 if n < num_in else 0 if n < num_in + num_core else 2) for n in range(num_nodes) }

	node_pairs = [(n1, n2) for n1 in range(num_nodes) 
		for n2 in (np.arange(n1 + 1, num_in + num_core) if n1 < num_in
		else np.arange(num_in, num_nodes) if n1 < num_in + num_core
		else np.arange(n1 + 1, num_nodes))]

	probs = [connection_probs[node_labels[n1], node_labels[n2]] for (n1, n2) in node_pairs]
	

	adj = np.zeros((num_nodes, num_nodes))
	adj[tuple(zip(*node_pairs))] = np.random.rand(len(probs)) < probs

	return node_labels, adj

def build_bow_tie_2(num_nodes, num_edges, kernel=lambda x : x, seed=0, ):
	
    def to_directed(graph, direction=lambda u, v: v >= u):
        if nx.is_directed(graph):
            return graph
        degrees = dict(nx.degree(graph))
        return nx.DiGraph([(u, v) if direction(degrees[u], degrees[v]) else (v, u) for u, v in graph.edges()])

    num_core = int(np.ceil(num_nodes/10.))
    num_nodes += num_core # account for overlap
    num_periphery_in = int(np.ceil(num_nodes/2))
    num_periphery_out = num_nodes - num_periphery_in

    # periphery_in = to_directed(nx.barabasi_albert_graph(num_periphery_in, 
    #     m=num_edges, seed=2*seed))
    periphery_in = nx.gn_graph(num_periphery_in, kernel=kernel, seed=2*seed)
    # periphery_in = nx.generators.directed.scale_free_graph(num_periphery_in, 
    	# alpha=0.46, beta=0.54, gamma=1e-10, seed=2*seed)

    periphery_in_degrees = dict(periphery_in.degree())
    periphery_in_nodes_sorted = sorted(periphery_in_degrees, key=periphery_in_degrees.get, reverse=True)
    periphery_in_core_nodes = periphery_in_nodes_sorted[:num_core]
    periphery_in_periphery_nodes = periphery_in_nodes_sorted[num_core:]

    # relabel core
    periphery_in = nx.relabel_nodes(periphery_in, {n: "core_{}".format(i) for i, n in enumerate(periphery_in_core_nodes)})
    # relabel periphery
    periphery_in = nx.relabel_nodes(periphery_in, {n: "periphery_in_{}".format(i) for i, n in enumerate(periphery_in_periphery_nodes)})

    # periphery_out = to_directed(nx.barabasi_albert_graph(num_periphery_out, 
        # m=num_edges, seed=2*seed+1), direction=lambda u, v: v <= u)
    periphery_out = nx.gn_graph(num_periphery_out, kernel=kernel, seed=2*seed+1).reverse()
    # periphery_out = nx.generators.directed.scale_free_graph(num_periphery_out, 
    	# alpha=0.46, beta=0.54, gamma=1e-10, seed=2*seed+1).reverse()

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

    node_labels = {v: (0 if "core" in k else 1 if "in" in k else 2) for k, v in mapping.items()}
    node_labels = pd.DataFrame.from_dict(node_labels, orient="index")

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
	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	edgelist_directory = args.edgelist_directory

	num_nodes = 400
	num_seeds = 30

	seed = args.seed
	core_prob = 1./6

	directory = os.path.join(edgelist_directory, "synthetic_bow_tie")
	if not os.path.exists(directory):
		print ("Making directory: {}".format(directory))
		os.makedirs(directory, exist_ok=True)

	for theta1 in np.arange(0.10, 1.0, 0.05):
		for theta2 in np.arange(0.05, theta1, 0.05):

			filename = "theta1={:.02f}-theta2={:.02f}-seed={:02d}".format(theta1, theta2, seed)
			edgelist_filename = os.path.join(directory, filename + ".edgelist")
			node_label_filename = os.path.join(directory, filename + ".csv")

			connection_probs = np.array([[theta1, 0,      theta1],
										 [theta1, theta2, 0],
										 [0,      0,      theta2]])

			# if os.path.exists(edgelist_filename):
			# 	print ("{} already exists".format(edgelist_filename))
			# 	return

			node_labels, adj = build_bow_tie(num_nodes, 
				core_prob, 
				connection_probs,
				seed=seed)
			graph = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
			graph.remove_edges_from(graph.selfloop_edges())
			# change direction to point "in" or "out"
			for u, v in list(graph.edges()):
			    if node_labels[u] == node_labels[v]:
			        if node_labels[u] == 1: # in component
			            if graph.degree(u) > graph.degree(v):
			                graph.remove_edge(u, v)
			                graph.add_edge(v, u)
			        elif node_labels[u] == 2: # out component
			            if graph.degree(v) > graph.degree(u):
			                graph.remove_edge(u, v)
			                graph.add_edge(v, u)
			nx.set_edge_attributes(graph, values=1, name="weight")
			nx.write_edgelist(graph, edgelist_filename, delimiter="\t")
			node_labels =  pd.DataFrame.from_dict(node_labels, orient="index")
			node_labels.to_csv(node_label_filename, sep=",")

			print ("Completed {}".format(filename))

			node_labels = node_labels.reindex(graph.nodes()).values.flatten()
			graph = nx.convert_node_labels_to_integers(graph)

			core = graph.subgraph([n for n in graph.nodes() if node_labels[n] == 0])
			periphery_in = graph.subgraph([n for n in graph.nodes() if node_labels[n] == 1])
			periphery_out = graph.subgraph([n for n in graph.nodes() if node_labels[n] == 2])

			for n1 in periphery_in.nodes():
				for n2 in periphery_out.nodes():
					assert not (n1, n2) in graph.edges() and not (n2, n1) in graph.edges()
			print("Passed")

			print ("Number of nodes = {}, number of edges = {}".format(len(graph), len(graph.edges())))
			print ("Network density = {}".format(nx.density(graph)))
			print ("Core density = {}".format(nx.density(core)))
			print ("Periphery in density = {}".format(nx.density(periphery_in)))
			print ("Periphery out density = {}".format(nx.density(periphery_out)))

if __name__ == "__main__":
	main()
