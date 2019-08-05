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
from networkx.drawing.nx_agraph import write_dot, graphviz_layout, to_agraph

import itertools

import random

def build_bow_tie(num_nodes, 
	num_core, 
	num_tendrils=4, 
	num_tubes=2, 
	core_p=.3,
	inhibitor_p=.5,
	tendril_length_range=[1,2,3],
	tube_length_range=[1,2,3],
	kernel=lambda x : x, 
	seed=0, ):

	np.random.seed(seed)
	random.seed(seed)

	num_periphery_in = int(np.ceil(num_nodes/2))
	num_periphery_in_core = int(np.ceil(num_core/2))
	num_periphery_out = num_nodes - num_periphery_in
	num_periphery_out_core = num_core - num_periphery_in_core

	periphery_in = nx.gn_graph(num_periphery_in, 
		kernel=kernel, seed=2*seed)

	periphery_in_degrees = dict(periphery_in.degree())
	periphery_in_nodes_sorted = sorted(periphery_in_degrees, key=periphery_in_degrees.get, reverse=True)
	periphery_in_core_nodes = periphery_in_nodes_sorted[:num_periphery_in_core]
	periphery_in_periphery_nodes = periphery_in_nodes_sorted[num_periphery_in_core:]

	nx.set_edge_attributes(periphery_in, name="weight", 
		values={edge: np.random.choice([-1, 1], p=[inhibitor_p, 1-inhibitor_p]) for edge in periphery_in.edges()})

	# # relabel core
	# periphery_in = nx.relabel_nodes(periphery_in, {n: "core_in_{}".format(i) for i, n in enumerate(periphery_in_core_nodes)})
	# # relabel periphery
	# periphery_in = nx.relabel_nodes(periphery_in, {n: "periphery_in_{}".format(i) for i, n in enumerate(periphery_in_periphery_nodes)})

	periphery_out = nx.gn_graph(num_periphery_out, 
		kernel=kernel, seed=2*seed+1).reverse()

	periphery_out_degrees = dict(periphery_out.degree())
	periphery_out_nodes_sorted = sorted(periphery_out_degrees, key=periphery_out_degrees.get, reverse=True)
	periphery_out_core_nodes = periphery_out_nodes_sorted[:num_periphery_out_core]
	periphery_out_periphery_nodes = periphery_out_nodes_sorted[num_periphery_out_core:]

	nx.set_edge_attributes(periphery_out, name="weight", 
		values={edge: np.random.choice([-1, 1], p=[inhibitor_p, 1-inhibitor_p]) for edge in periphery_out.edges()})

	# relabel core
	# periphery_out = nx.relabel_nodes(periphery_out, {n: "core_out_{}".format(i) for i, n in enumerate(periphery_out_core_nodes)})
	# # relabel periphery
	# periphery_out = nx.relabel_nodes(periphery_out, {n: "periphery_out_{}".format(i) for i, n in enumerate(periphery_out_periphery_nodes)})

	graph = nx.union(periphery_in, periphery_out, rename=("in-", "out-"))

	# add more edges to core
	core = graph.subgraph(["in-{}".format(i) 
		for i in periphery_in_core_nodes] + ["out-{}".format(i) for i in periphery_out_core_nodes])

	## self edges in core
	for u in sorted(core.nodes()):
		if np.random.rand() < core_p:
			graph.add_edge(u, u, weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))

	for u, v in itertools.combinations(sorted(core.nodes()), 2):
		if np.random.rand() < core_p:
			graph.add_edge(u, v, weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))
		if np.random.rand() < core_p:
			graph.add_edge(v, u, weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))

	while not nx.is_strongly_connected(core):
		edge = np.random.choice(core, size=2, replace=False)
		graph.add_edge(edge[0], edge[1], weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))
	
	graph = nx.convert_node_labels_to_integers(graph,)

	core = set(max(nx.strongly_connected_component_subgraphs(graph), key=len))
	periphery_in = set()
	periphery_out = set()
	for n in core:
		for u in nx.ancestors(graph, n) - core:
			assert u not in periphery_out
			periphery_in.add(u)
		for u in nx.descendants(graph, n) - core:
			assert u not in periphery_in
			periphery_out.add(u)

	periphery_in = list(periphery_in)
	periphery_out = list(periphery_out)

	# add tendrils
	tendril_nodes = []
	for i in range(num_tendrils):
		tendril_length = np.random.choice(tendril_length_range)  
		# even -- start from in
		if i % 2 == 0:

			start = np.random.choice(periphery_in)
			graph.add_edge(start, len(graph), weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))
			tendril_nodes.append(len(graph) - 1)    
			for j in range(tendril_length):
				graph.add_edge(len(graph) - 1, len(graph), weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))
				tendril_nodes.append(len(graph) - 1)

		else: # odd -- end at out component

			graph.add_node(len(graph))
			tendril_nodes.append(len(graph) - 1)

			for j in range(tendril_length):
				graph.add_edge(len(graph) -1, len(graph), weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))
				tendril_nodes.append(len(graph) - 1)

			end = np.random.choice(periphery_out)
			graph.add_edge(len(graph) - 1, end, weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))

	# add tubes
	tube_nodes = []
	for i in range(num_tubes):
		# start from in
		start = np.random.choice(periphery_in)
		# end at out component
		end = np.random.choice(periphery_out)
		tube_length = np.random.choice(tube_length_range)

		graph.add_edge(start, len(graph), weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))
		tube_nodes.append(len(graph) - 1)

		for j in range(tube_length):
			graph.add_edge(len(graph) - 1, len(graph), weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))
			tube_nodes.append(len(graph) - 1)
		graph.add_edge(len(graph) - 1, end, weight=np.random.choice([-1., 1.], p=[inhibitor_p, 1-inhibitor_p]))

	# 0 core
	# 1 in
	# 2 out
	# 3 tendril
	# 4 tube

	node_labels = np.ones(len(graph), dtype=int) * 3
	node_labels[list(core)] = 0
	node_labels[periphery_in] = 1
	node_labels[periphery_out] = 2
	node_labels[tendril_nodes] = 3
	node_labels[tube_nodes] = 4

	return node_labels, graph

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Run core-periphery detections algorithms")

	parser.add_argument("--edgelist_dir", dest="edgelist_directory", type=str, default="datasets/",
		help="The directory containing edgelist files (default is 'datasets/').")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	edgelist_directory = args.edgelist_directory

	num_nodes = 30
	num_core = 4
	num_seeds = 5

	root_directory = os.path.join(edgelist_directory, "synthetic_bow_tie")
	if not os.path.exists(root_directory):
		print ("Making directory: {}".format(root_directory))
		os.makedirs(root_directory, exist_ok=True)

	for seed in range(num_seeds):
		# seed = args.seed

		directory = os.path.join(root_directory, "seed={:03d}".format(seed))

		if not os.path.exists(directory):
			print ("Making directory: {}".format(directory))
			os.makedirs(directory, exist_ok=True)

		edgelist_filename = os.path.join(directory, "edgelist.tsv")
		condensed_edgelist_filename = os.path.join(directory, "edgelist_condensed.tsv")

		node_label_filename = os.path.join(directory, "labels.csv")
		condensed_map_filename = os.path.join(directory, "condensed_map.csv")

		node_labels, graph = build_bow_tie(num_nodes, num_core, 
			num_tendrils=4, num_tubes=2,
			core_p=.3,
			inhibitor_p=0.5,
			kernel=lambda x: x**.75, seed=seed)  

		assert nx.is_weakly_connected(graph)

		nx.set_edge_attributes(graph, name="arrowhead",
			values={(u, v): ("normal" if w==1. else "tee") 
				for u, v, w in graph.edges(data="weight")})

		colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
		nx.set_node_attributes(graph, name="color", 
			values={n: colors[node_labels[n]] for n in graph.nodes()})

		# core = max(nx.strongly_connected_component_subgraphs(graph, copy=False), key=len)

		# randomly assign edge labels to core
		# nx.set_edge_attributes(core, name="weight", values={(u, v): np.random.choice([-1., 1.]) for u, v in core.edges()})
		# nx.write_edgelist(core, os.path.join(directory, "core.tsv"), delimiter="\t", data=["weight"])

		# ensure that every node appears in edgelist at least once with dummy self edges
		# graph.add_weighted_edges_from((u, u, 0.) for u in sorted(graph.nodes()) if (u, u) not in graph.edges())

		nx.write_edgelist(graph, edgelist_filename, delimiter="\t", data=["weight"])

		network_plot_filename = os.path.join(directory, "whole_network.png")
		graph.graph['edge'] = {'arrowsize': '.8', 'splines': 'curved'}
		graph.graph['graph'] = {'scale': '3'}
		
		a = to_agraph(graph)
		a.layout('dot')   
		a.draw(network_plot_filename)
		

		# node_labels = pd.DataFrame(node_labels)
		pd.DataFrame(node_labels).to_csv(node_label_filename, sep=",")

		graph_condensed = nx.condensation(graph)
		nx.set_edge_attributes(graph_condensed, name="weight", values=1.)
		nx.write_edgelist(graph_condensed, condensed_edgelist_filename, delimiter="\t", data=["weight"])

		members = nx.get_node_attributes(graph_condensed, "members")
		core_node = max(members, key= lambda x: len(set(members[x])))

		in_component = list(nx.ancestors(graph_condensed, core_node))
		out_component = list(nx.descendants(graph_condensed, core_node))

		condensed_node_map = {n_: n for n, l in nx.get_node_attributes(graph_condensed, name="members").items() for n_ in l}
	   
		assert len(condensed_node_map) == len(graph)
		pd.DataFrame.from_dict(condensed_node_map, orient="index").to_csv(condensed_map_filename) 

		inverse_map = {v: k for k, v in condensed_node_map.items()}
		graph_condensed_labels = np.array([node_labels[inverse_map[n]] for n in sorted(graph_condensed.nodes())])
		
		for u in in_component:
			assert graph_condensed_labels[u] == 1
		for u in out_component:
			assert graph_condensed_labels[u] == 2

		pd.DataFrame(graph_condensed_labels).to_csv(os.path.join(directory, "labels_condensed.csv"), sep=",")

		nx.set_node_attributes(graph_condensed, name="color", 
			values={n: colors[graph_condensed_labels[n]] for n in graph_condensed.nodes()})
		
		network_plot_filename = os.path.join(directory, "condensed_network.png")
		graph_condensed.graph['edge'] = {'arrowsize': '.8', 'splines': 'curved'}
		graph_condensed.graph['graph'] = {'scale': '3'}
		
		a = to_agraph(graph_condensed)
		a.layout('dot')   
		a.draw(network_plot_filename)
		
		print ("Completed seed={}".format(seed))

if __name__ == "__main__":
	main()
