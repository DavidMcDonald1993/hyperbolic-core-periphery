import os

import numpy as np
import networkx as nx

import argparse
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout, to_agraph
from networkx.algorithms.dag import descendants


import itertools
import glob

def hamming_distance(y_true, y_pred, axis=-1):
	return np.abs(y_true - y_pred).mean(axis=axis).mean()

def synchronous_update(A, p, theta=0.):
	p = p.copy()

	Ap  = A.dot(p)

	p[Ap > theta] = 1
	p[Ap < theta] = 0

	return p

def simulate_dynamics(A, p, theta=0.):

	iter_ = 0
	
	states = []

	while True:

		p = synchronous_update(A, p)

		for i, state in enumerate(states[::-1]):
			if np.allclose(p, state):
				if i == 0:
					print ("steady state at iteration", iter_)
				else:
					print ("oscillation of", i, "states found at iteration", iter_)
				return p

		states.append(p.copy())

		iter_ += 1

def measure_perturbation(A, perturbation_size=3, theta=0., 
	num_initial_conditions=50, max_iters=1000):
	
	N = len(A)

	assert perturbation_size < N

	p = np.random.randint(2, size=(N, num_initial_conditions))
	p_ = p.copy()
	for _p in p_.T:
		idx = np.random.choice(N, size=perturbation_size, replace=False)
		_p[idx] = 1 - _p[idx]
	assert not np.allclose(p, p_)

	hd = hamming_distance(p, p_, axis=0)

	hds = [hd]
	for iter_ in range(max_iters):

		p = synchronous_update(A, p)
		p_ = synchronous_update(A, p_)
		hd = hamming_distance(p, p_, axis=0)
		hds.append(hd)

	return hds

def read_cycles(filename):
	print ("reading cycles from", filename)

	cycles = []
	with open(filename, "r") as f:

		for line in (line.rstrip() for line in f.readlines()):

			cycle = tuple(map(int, line.split(",")))
			print ("read cycle:", cycle)
			cycles.append(cycle)

	return cycles

def construct_cycle_subgraphs(cycles, core):
	core_adj = nx.adjacency_matrix(core).A 
	map_ = {n: i for i, n in enumerate(core.nodes())}
	cycle_subgraphs = []#[core_adj]
	for cycle in cycles:
		cycle_adj = np.zeros_like(core_adj)
		cycle = [map_[n] for n in cycle]
		cycle_adj[cycle, cycle[1:] + cycle[:1]] = \
		core_adj[cycle, cycle[1:] + cycle[:1]]

		cycle_subgraphs.append(cycle_adj.T)

	return cycle_subgraphs

# def evaluate(n, p, cycle_tree, cycle_subgraphs, theta=0.):

# 	adj = cycle_subgraphs[n]
# 	children = list(cycle_tree.neighbors(n))
# 	p_children = np.array([evaluate(child, p, cycle_tree, cycle_subgraphs, theta=theta) 
# 		for child in children]).sum(axis=0)
# 	if n != 0:
# 		p_n = adj.dot(p) + p_children
# 	else:
# 		p_n = p.copy()
# 		p_n[p_children > theta] = 1
# 		p_n[p_children < theta] = 0

# 	return p_n

def attractor_landscape(core_adj, theta=0.):
	assert isinstance(core_adj, np.ndarray)
	assert len(core_adj) < 16
	p = np.array(list(itertools.product([0, 1], repeat=len(core_adj)))).T
	num_states = p.shape[1]

	print ("number of possible states =", num_states)

	map_ = {tuple(a): i for i, a in enumerate(p.T)}

	landscape = nx.DiGraph()

	# one iteration is sufficient
	p_ = synchronous_update(core_adj, p)

	for p1, p2 in zip(p.T, p_.T):
		p1 = tuple(p1)
		p2 = tuple(p2)

		if p2 not in map_:
			map_.update({p2: len(map_)})

		u = map_[p1]
		v = map_[p2]

		landscape.add_edge(u, v)

	p = p_

	print ("number of attractors =", nx.number_weakly_connected_components(landscape))
	return landscape

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Deconstruct core into cycle parent tree")

	parser.add_argument("-d", "--dir", dest="output_dir", type=str, 
		help="The directory to write cycles and parent map to.")
	args = parser.parse_args()
	return args

def main():

	np.random.seed(0)

	args = parse_args()

	for core_no, core_edgelist in enumerate(glob.iglob(os.path.join(args.output_dir, "core_*.tsv"))):

		core = nx.read_weighted_edgelist(core_edgelist, create_using=nx.DiGraph(), nodetype=int)

		# print (core.nodes())
		print ("core number", core_no)
		print ("number of nodes in core:", len(core))
		print ("number of edges in core:", len(core.edges()))

		core_adj = nx.adjacency_matrix(core).A.T
		print ("CORE ADJ")
		print (core_adj)

		assert nx.is_strongly_connected(core)

		attractor_landscape_filename = os.path.join(args.output_dir, 
			"attractor_landscape_{}.png".format(core_no))
		# if not os.path.exists(attractor_landscape_filename):
			
		landscape = attractor_landscape(core_adj)

		landscape.graph['edge'] = {'arrowsize': '.3', 'splines': 'curved'}
		landscape.graph['graph'] = {'scale': '3'}
		a = to_agraph(landscape)
		a.layout('dot')   
		a.draw(attractor_landscape_filename)

		cycles = read_cycles(os.path.join(args.output_dir, "cycles_{}.csv".format(core_no)))

		cycle_tree = nx.read_weighted_edgelist(os.path.join(args.output_dir, "cycle_tree_{}.tsv".format(core_no)),
			delimiter="\t", 
			create_using=nx.DiGraph(), 
			nodetype=int)

		cycle_tree = cycle_tree.reverse() 

		cycle_subgraphs = construct_cycle_subgraphs(cycles, 
			core)
		cycle_subgraphs = np.array(cycle_subgraphs)
		
		cycle_subgraphs =  cycle_subgraphs /\
		(np.abs(cycle_subgraphs).sum(axis=0, keepdims=True) + 1e-15)

		cycle_subgraphs = np.array([cycle_subgraph /\
			len(list(nx.algorithms.simple_paths.all_simple_paths(cycle_tree,
			0, n+1)))
			for n, cycle_subgraph 
			in enumerate(cycle_subgraphs)])

		cycle_subgraphs = np.concatenate( (np.expand_dims(np.identity(len(core)), 0), cycle_subgraphs), axis=0 )

		assert len(core) < 16
		p = np.array(list(itertools.product([0, 1], repeat=len(core)))).T

		theta = -0.

		p_core = synchronous_update(core_adj, 
			p, theta=theta)

		d = {n: descendants(cycle_tree, n) for n in cycle_tree}
		ordering = sorted(d, key=lambda x: len(d[x]))

		assert ordering[-1] == 0

		expressions = {}
		hamming_distances = {}

		for n in ordering:
			adj = cycle_subgraphs[n]

			p_ = synchronous_update(adj, p)

			mask = np.any(adj, axis=1, keepdims=True)

			# mask = np.ones_like(p)
			hamming_distances.update({n: hamming_distance(mask*p_core, 
				mask*p_, axis=0)})

			neighbours = cycle_tree.neighbors(n)
			p_children = np.array([expressions[n] 
				for n in neighbours]).sum(axis=0)
			if isinstance(p_children, np.ndarray):
				p_children[np.abs(p_children) < 1e-7] = 0
			if n != 0:
				p_n = adj.dot(p) + p_children
			else:
				p_n = p.copy()
				p_n[p_children > theta] = 1
				p_n[p_children < theta] = 0

			expressions[n] = p_n

		print ("HAMMING DISTANCES")
		print (hamming_distances)

		p_tree = expressions[0]

		print ("ORIGINAL")
		print (p)
		print ("CORE")
		print (p_core)
		print ("TREE")
		print (p_tree)
		print ((p_tree - p_core).sum())

if __name__ == "__main__":
	main()