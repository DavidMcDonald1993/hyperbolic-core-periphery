import numpy as np
import networkx as nx
import pandas as pd

import os
import argparse

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout, to_agraph

from visualise import plot_degree_dist

# def determine_backedges(core, tree):
# 	descendants = {u: set([u] + list(nx.descendants(tree, u))) for u in core}
# 	back_edges = {n: set() for n in core.nodes()}
# 	for (u, v) in core.edges():
# 		if u in descendants[v]:
# 			back_edges[u].add(v)
# 	return back_edges

# def directed_chain_decomposition(core, tree, root):

# 	dfs_nodes = nx.dfs_preorder_nodes(core.reverse(), source=root)
# 	back_edges = determine_backedges(core, tree)

# 	visited = {n: False for n in core.nodes()}

# 	chains = []

# 	for n in dfs_nodes:
# 	    visited[n] = True
# 	    for v in sorted(back_edges[n], key=lambda v_: nx.shortest_path_length(tree, source=v_, target=n), reverse=True):
# 	        u = n
# 	        chain = [(u, v)]
# 	        while not visited[v]:
# 	            visited[u] = True
# 	            visited[v] = True
# 	            u = v
# 	            v = list(tree.neighbors(u))[0]
# 	            assert (u, v) in tree.edges()
# 	            chain.append((u, v))
# 	        chains.append(chain)

# 	return chains

def chain_decomposition(G, root=None):
	"""Returns the chain decomposition of a graph.

	The *chain decomposition* of a graph with respect a depth-first
	search tree is a set of cycles or paths derived from the set of
	fundamental cycles of the tree in the following manner. Consider
	each fundamental cycle with respect to the given tree, represented
	as a list of edges beginning with the nontree edge oriented away
	from the root of the tree. For each fundamental cycle, if it
	overlaps with any previous fundamental cycle, just take the initial
	non-overlapping segment, which is a path instead of a cycle. Each
	cycle or path is called a *chain*. For more information, see [1]_.

	Parameters
	----------
	G : undirected graph

	root : node (optional)
	   A node in the graph `G`. If specified, only the chain
	   decomposition for the connected component containing this node
	   will be returned. This node indicates the root of the depth-first
	   search tree.

	Yields
	------
	chain : list
	   A list of edges representing a chain. There is no guarantee on
	   the orientation of the edges in each chain (for example, if a
	   chain includes the edge joining nodes 1 and 2, the chain may
	   include either (1, 2) or (2, 1)).

	Raises
	------
	NodeNotFound
	   If `root` is not in the graph `G`.

	Notes
	-----
	The worst-case running time of this implementation is linear in the
	number of nodes and number of edges [1]_.

	References
	----------
	.. [1] Jens M. Schmidt (2013). "A simple test on 2-vertex-
	   and 2-edge-connectivity." *Information Processing Letters*,
	   113, 241–244. Elsevier. <https://doi.org/10.1016/j.ipl.2013.01.016>

	"""

	def _dfs_cycle_forest(G, root=None):
		"""Builds a directed graph composed of cycles from the given graph.

		`G` is an undirected simple graph. `root` is a node in the graph
		from which the depth-first search is started.

		This function returns both the depth-first search cycle graph
		(as a :class:`~networkx.DiGraph`) and the list of nodes in
		depth-first preorder. The depth-first search cycle graph is a
		directed graph whose edges are the edges of `G` oriented toward
		the root if the edge is a tree edge and away from the root if
		the edge is a non-tree edge. If `root` is not specified, this
		performs a depth-first search on each connected component of `G`
		and returns a directed forest instead.

		If `root` is not in the graph, this raises :exc:`KeyError`.

		"""
		# Create a directed graph from the depth-first search tree with
		# root node `root` in which tree edges are directed toward the
		# root and nontree edges are directed away from the root. For
		# each node with an incident nontree edge, this creates a
		# directed cycle starting with the nontree edge and returning to
		# that node.
		#
		# The `parent` node attribute stores the parent of each node in
		# the DFS tree. The `nontree` edge attribute indicates whether
		# the edge is a tree edge or a nontree edge.
		#
		# We also store the order of the nodes found in the depth-first
		# search in the `nodes` list.
		H = nx.DiGraph()
		nodes = []
		for u, v, d in nx.dfs_labeled_edges(G, source=root):
			if d == 'forward':
				# `dfs_labeled_edges()` yields (root, root, 'forward')
				# if it is beginning the search on a new connected
				# component.
				if u == v:
					H.add_node(u, parent=None)
					nodes.append(u)
				else:
					H.add_node(v, parent=u)
					H.add_edge(u, v, nontree=False)
					nodes.append(v)
			# `dfs_labeled_edges` considers nontree edges in both
			# orientations, so we need to not add the edge if it its
			# other orientation has been added.
			elif d == 'nontree' and v not in H[u]:
				H.add_edge(u, v, nontree=True)
			else:
				# Do nothing on 'reverse' edges; we only care about
				# forward and nontree edges.
				pass
		for edge in H.edges:
			if edge[0]==edge[1]:
				continue
			assert edge in G.edges
		return H.reverse(), nodes

	def _build_chain(G, u, v, visited):
		"""Generate the chain starting from the given nontree edge.

		`G` is a DFS cycle graph as constructed by
		:func:`_dfs_cycle_graph`. The edge (`u`, `v`) is a nontree edge
		that begins a chain. `visited` is a set representing the nodes
		in `G` that have already been visited.

		This function yields the edges in an initial segment of the
		fundamental cycle of `G` starting with the nontree edge (`u`,
		`v`) that includes all the edges up until the first node that
		appears in `visited`. The tree edges are given by the 'parent'
		node attribute. The `visited` set is updated to add each node in
		an edge yielded by this function.

		"""
		while v not in visited:
			assert (u, v) in G.edges()
			yield u, v
			visited.add(v)
			for u, v, d in G.out_edges(v, data="nontree"):
				if not d:
					break
		yield u, v

	# Create a directed version of H that has the DFS edges directed
	# toward the root and the nontree edges directed away from the root
	# (in each connected component).
	H, nodes = _dfs_cycle_forest(G.reverse(), root)
	
	assert (len(G.edges) == len(H.edges)), (len(G.edges), len(H.edges))
	
	for edge in H.edges():
		if edge[0]==edge[1]:
			continue
		assert edge in G.edges(), edge

	# Visit the nodes again in DFS order. For each node, and for each
	# nontree edge leaving that node, compute the fundamental cycle for
	# that nontree edge starting with that edge. If the fundamental
	# cycle overlaps with any visited nodes, just take the prefix of the
	# cycle up to the point of visited nodes.
	#
	# We repeat this process for each connected component (implicitly,
	# since `nodes` already has a list of the nodes grouped by connected
	# component).
	visited = set()
	chains = []
	for u in nodes:
		visited.add(u)
		# For each nontree edge going out of node u...
		edges = ((u, v) for u, v, d in H.out_edges(u, data='nontree') if d)
		for u, v in edges:
			# Create the cycle or cycle prefix starting with the
			# nontree edge.
			chain = list(_build_chain(H, u, v, visited))
			chains.append(chain)
	return chains, H

def construct_s_belongs_dict(chains, root, tree):
	s_belongs = {root: 0}
	for i, chain in enumerate(chains):
		for u, v in chain:
			if (u, v, False) in tree.edges(data="nontree"): # u, P(u)
				s_belongs.update({u: i})
	return s_belongs

def build_parent_map(chains, tree, s_belongs):
	parent_map = nx.DiGraph()

	for i, chain in enumerate(chains):
		if i == 0:
			continue
		s_chain = chain[0][0]
		t_chain = chain[-1][-1]    
		path = nx.shortest_path(tree, t_chain, s_chain)
		parent_chain_idx = s_belongs[t_chain]
		parent_chain = chains[parent_chain_idx]
		s_parent_chain = parent_chain[0][0]
		t_parent_chain = parent_chain[-1][-1]   

		parent_map.add_edge(i, parent_chain_idx)

	return parent_map

class Tree(object):
	
	def __init__(self, value, i, depth):
		self.value = value
		self.i = i
		self.depth = depth
		self.children = []

	def overlap(self, s1, s2):
		return len(s1.intersection(s2)) / len(s1)
	
	def insert(self, value, i):
		
		inserted = False
		# max_overlap = 0
		# idx = -1


		for idx_, child in enumerate(self.children):
			# overlap = self.overlap(child.value, value)
			# if overlap > max_overlap:
			# 	max_overlap = overlap
			# 	idx = idx_
			if set(value) < set(child.value):
				child.insert(value, i)
				inserted = True
				# break
		if not inserted:
		# if max_overlap > 0:
			# self.children[idx].insert(value, i)
		# else:
			self.children.append(Tree(value, i, self.depth+1))
		
	def __str__ (self):
		s = "-" * self.depth + str(self.i) + " " + str(self.value) 
		for child in self.children:
			s += "\n" + str(child)
		return s

def tree_to_nx(tree):

	cycle_tree = nx.DiGraph()
	l = [tree]
	while len(l) > 0:
		t_ = l.pop(0)
		
		for child in t_.children:
			cycle_tree.add_edge(child.i, t_.i)
			l.append(child)

	return cycle_tree

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Deconstruct core into cycle parent tree")

	# parser.add_argument("--edgelist", dest="edgelist", type=str, 
	# 	help="The edgelist filename.")
	parser.add_argument("-d", "--dir", dest="output_dir", type=str, 
		help="The directory to write chains and parent map to.")
	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	edgelist = os.path.join(args.output_dir, "edgelist.tsv")
	assert edgelist.endswith(".tsv") # accept whole graph or just core

	graph = nx.read_weighted_edgelist(edgelist, create_using=nx.DiGraph(), nodetype=int)

	graph = max(nx.weakly_connected_component_subgraphs(graph, copy=False), key=len) # focus on largest WCC

	# core = max(nx.strongly_connected_component_subgraphs(graph, copy=False), key=len)
	for core_no, core in enumerate(filter(lambda x: len(x) > 1, nx.strongly_connected_component_subgraphs(graph))):

		print ("core no:", core_no)
		print ("number of nodes in core:", len(core))
		print ("number of edges in core:", len(core.edges()))

		core_plot_filename = os.path.join(args.output_dir, "core_{}.png".format(core_no))
		# if not os.path.exists(core_plot_filename):
		gene_df_filename = os.path.join(args.output_dir, "gene_ids.csv")
		if os.path.exists(gene_df_filename):
			map_ = pd.read_csv(gene_df_filename, header=None, index_col=0)[1].to_dict()
			_core = nx.relabel_nodes(core, mapping=map_)
		else: 
			_core = core
		_core.graph['edge'] = {'arrowsize': '.8', 'splines': 'curved'}
		_core.graph['graph'] = {'scale': '3'}
		nx.set_edge_attributes(_core, name="arrowhead",
			values={(u, v): ("normal" if w==1 else "tee") 
				for u, v, w in _core.edges(data="weight")})
		a = to_agraph(_core)
		a.layout('dot')   
		a.draw(core_plot_filename)

		core_edgelist_filename = os.path.join(args.output_dir, "core_{}.tsv".format(core_no))
		nx.write_edgelist(core, core_edgelist_filename,
		delimiter="\t", data=["weight"])

		print ("wrote core edgelist to", core_edgelist_filename)

		assert nx.is_strongly_connected(core)

		# # identify all cycles and sort by number of nodes
		# print ("identifying all cycles in core")
		# all_cycles = list(nx.cycles.simple_cycles(core))
		# print ("sorting cycles by number of nodes")
		# all_cycles = sorted(all_cycles, key=len, reverse=True)
		# print ("done")

		# # write chains
		# cycle_filename = os.path.join(args.output_dir, "cycles_{}.csv".format(core_no))
		# print ("writing cycles to:", cycle_filename)
		# with open(cycle_filename, "w",) as f:
		# 	for i, cycle in enumerate(all_cycles):
		# 		print (i+1, cycle)
		# 		for u in cycle[:-1]:
		# 			f.write("{},".format(u, ))
		# 		f.write("{}\n".format(cycle[-1], ))

		# # add all nodes in core to be root of tree
		# all_cycles = [list(core.nodes())] + all_cycles

		# # convert cycles to sets of nodes
		# all_cycles = list(map(set, all_cycles))

		# # root of tree
		# root = all_cycles.pop(0)

		# cycle_tree = Tree(root, 0, 0)

		# for i, cycle in enumerate(all_cycles):
		# 	print ("inserting", cycle, "into tree.")
		# 	cycle_tree.insert(cycle, i+1)

		# print ("Produced tree:")
		# print (cycle_tree)

		# print ("converting to networkx")
		# cycle_tree = tree_to_nx(cycle_tree)

		root = list(core.nodes())[0] # arbitrary root?

		chains, tree = chain_decomposition(core, root=root)

		for chain in chains:
			print (chain)

		s_belongs = construct_s_belongs_dict(chains, root, tree)

		print (s_belongs)

		cycle_tree = build_parent_map(chains, tree, s_belongs)

		print ("plotting")
		cycle_tree.graph['edge'] = {'arrowsize': '.3', 'splines': 'curved'}
		cycle_tree.graph['graph'] = {'scale': '3'}
		a = to_agraph(cycle_tree)
		a.layout('dot')   
		a.draw(os.path.join(args.output_dir, "cycle_tree_{}.png".format(core_no)))

		cycle_tree_filename = os.path.join(args.output_dir, "cycle_tree_{}.tsv".format(core_no))
		print ("writing cycle tree to:", cycle_tree_filename)
		nx.write_edgelist(cycle_tree, cycle_tree_filename, delimiter="\t", data=["weight"])

if __name__ == "__main__":
	main()