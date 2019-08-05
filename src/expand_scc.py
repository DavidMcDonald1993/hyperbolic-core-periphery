import numpy as np
import networkx as nx
import pandas as pd

import os
import argparse

import matplotlib.pyplot as plt

from utils import load_data
from visualise import plot_degree_dist

def determine_nodes_in_core(chains):
	nodes_in_core = set()
	for chain in chains:
		for u, v in chain:
			nodes_in_core.add(u)
			nodes_in_core.add(v)
	return nodes_in_core

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Replace core into chain parent tree")

	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="The edgelist filename.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="The labels filename.")
	parser.add_argument("--parent_tree_dir", dest="parent_tree_dir", type=str, 
		help="The directory to read chains and parent map from.")
	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	graph = nx.read_weighted_edgelist(args.edgelist, delimiter="\t", nodetype=int, create_using=nx.DiGraph())

	assert all(u in graph for u in range(len(graph)))
	
	# select largest wcc
	print (len(graph), len(graph.edges()))
	graph = max(nx.weakly_connected_component_subgraphs(graph), key=len)
	print (len(graph), len(graph.edges()))

	plot_degree_dist(graph, "original")

	scc = max(nx.strongly_connected_component_subgraphs(graph), key=len)

	graph_condensed = nx.condensation(graph)

	plot_degree_dist(graph_condensed, "condensed")


	map_ = {v_: k for k, v in graph_condensed.nodes(data="members") for v_ in v}
	map_inv = {k: list(v)[0] for k, v in graph_condensed.nodes(data="members")}

	for u, v, w in graph.edges(data="weight"):
		if (map_[u], map_[v]) in graph_condensed.edges():
			graph_condensed.add_edge(map_[u], map_[v], weight=w)

	members = nx.get_node_attributes(graph_condensed, "members")
	core_node = max(members, key= lambda x: len(set(members[x])))

	# in_component = list(nx.ancestors(graph_condensed, core_node))
	# out_component = list(nx.descendants(graph_condensed, core_node))
	
	labels = pd.read_csv(args.labels, index_col=0, sep=",")
	labels = labels.reindex(sorted(graph.nodes())).values.flatten()
	assert len(labels.shape) == 1

	graph_condensed_labels = np.array([labels[map_inv[n]] for n in sorted(graph_condensed.nodes())])
	# graph_condensed_labels = -np.ones(len(graph_condensed), dtype=int)
	# graph_condensed_labels[core_node] = 0
	# graph_condensed_labels[in_component] = 1
	# graph_condensed_labels[out_component] = 2

	sum_weights_within_scc = sum((w for u, v, w in scc.edges(data="weight")))
	sum_weights_to_scc = {n : sum((d["weight"] for u, d in graph[map_inv[n]].items() if u in scc)) for n in in_component }
	graph_reversed = graph.reverse()
	sum_weights_from_scc = {n : sum((d["weight"] for u, d in graph_reversed[map_inv[n]].items() if u in scc)) for n in out_component }

	graph_condensed.add_edge(core_node, core_node, weight=sum_weights_within_scc)
	for n, w in sum_weights_to_scc.items():
		graph_condensed.add_edge(n, core_node, weight=w)
	for n, w in sum_weights_from_scc.items():
		graph_condensed.add_edge(core_node, n, weight=w)


	for u, v, w in graph_condensed.edges(data="weight"):
		assert w is not None

	graph_condensed.add_weighted_edges_from(((u, u, 0.) for u in graph_condensed))
	assert all(u in graph_condensed for u in range(len(graph_condensed)))

	nx.write_edgelist(graph_condensed, os.path.join(args.parent_tree_dir, "edgelist_condensed.tsv"), delimiter="\t", data=["weight"])
	pd.DataFrame(graph_condensed_labels).to_csv(os.path.join(args.parent_tree_dir, "labels_condensed.csv"))

	# pos = nx.spring_layout(graph)
	# nx.draw_networkx_nodes(graph, pos=pos)
	# nx.draw_networkx_edges(graph, pos=pos)
	# nx.draw_networkx_labels(graph, pos=pos)
	# plt.show()

	plot_degree_dist(graph_condensed, "condensed_reweighted")

	chains = []
	chain_filename = os.path.join(args.parent_tree_dir, "chains.csv", )
	# counts_chains = np.zeros(len(graph))
	counts_chains = {n: 0 for n in scc}
	with open(chain_filename, "r") as f:
		for line in (line.rstrip() for line in f.readlines()):
			chain = []
			edges = line.split("\t")
			for edge in edges:
				u, v = map(int, edge.split(","))
				chain.append((u, v))
				counts_chains[u] += 1
			chains.append(chain)

	# detemine nodes in core and all edges cpnnecting core and periphery
	# nodes_in_core = determine_nodes_in_core(chains)

	in_edge_dict = {n: [] for n in scc }
	out_edge_dict = {n: [] for n in scc }

	# counts_edges = np.zeros(len(graph))
	for u, v, w in graph.edges(data="weight"):
		if graph_condensed_labels[map_[u]] == 1 and v in scc:	
			in_edge_dict[v].append((map_[u], w))
			# counts_edges[v] += w
		if graph_condensed_labels[map_[v]] == 2 and u in scc:
			out_edge_dict[u].append((map_[v], w))
			# counts_edges[u] += w

	# weights = counts_edges / (counts_chains + 1e-15)

	# load parent map
	parent_map_filename = os.path.join(args.parent_tree_dir, "parent_map.tsv")
	parent_map = nx.read_weighted_edgelist(parent_map_filename, delimiter="\t", nodetype=int, 
									  create_using=nx.DiGraph())

	# replace core with parent map
	graph_expanded = graph_condensed.copy()
	graph_expanded.remove_node(core_node)

	parent_tree_map = {0: core_node}
	parent_tree_map.update({n: n+len(graph_condensed)-1 for n in range(1, len(parent_map))})

	for u, v, w in parent_map.edges(data="weight"):
		# add edge to g_expanded
		graph_expanded.add_edge(parent_tree_map[u], parent_tree_map[v], weight=w)

	plot_degree_dist(graph_expanded, "expanded_only_parent_tree")

	# connect core to periphery
	for u in parent_map.nodes():
		u_ = parent_tree_map[u]
		chain = chains[u]
		# if parent_map.in_degree(u) > 0:
		#     continue
		for node, _ in chain:
			# if chain_classifications[u] == "out":
			for out_node, w in out_edge_dict[node]:
				graph_expanded.add_edge(u_, out_node, weight=w/counts_chains[node])
			# else:
			for in_node, w in in_edge_dict[node]:
				graph_expanded.add_edge(in_node, u_, weight=w/counts_chains[node])

	for u, v, w in graph_condensed.edges(data="weight"):
		assert w is not None

	plot_degree_dist(graph_expanded, "expanded")

	expanded_edgelist_filename = os.path.join(args.parent_tree_dir, "edgelist_expanded.tsv")
	print ("writing expanded graph to:", expanded_edgelist_filename)
	graph_expanded.add_weighted_edges_from(((u, u, 0.) for u in graph_expanded))
	assert all(u in graph_expanded for u in range(len(graph_expanded)))
	nx.write_edgelist(graph_expanded, expanded_edgelist_filename, delimiter="\t", data=["weight"])

	# label ``outer core'' nodes as 0
	expanded_graph_labels = np.append(graph_condensed_labels, np.zeros(len(graph_expanded) - len(graph_condensed)) ).astype(int)
	expanded_graph_labels_filename = os.path.join(args.parent_tree_dir, "labels_expanded.csv")
	print ("saving expanded graph labels to:", expanded_graph_labels_filename)
	pd.DataFrame(expanded_graph_labels).to_csv(expanded_graph_labels_filename)





if __name__ == "__main__":
	main()