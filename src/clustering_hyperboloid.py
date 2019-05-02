from __future__ import print_function

import os
import re
import argparse
import numpy as np
import networkx as nx
import pandas as pd

from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt


from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import roc_auc_score


from utils import load_data
from visualise import draw_graph

def minkowki_dot(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance(u, v):
	mink_dp = minkowki_dot(u, v)
	mink_dp = np.minimum(mink_dp, -(1 + 1e-32))
	return np.arccosh(-mink_dp)

def perform_clustering(dists, eps):
	dbsc = DBSCAN(metric="precomputed", eps=eps, 
		n_jobs=-1, 
		min_samples=3).fit(dists)
	return dbsc.labels_, dbsc.core_sample_indices_

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def convert_module_to_directed_module(module, ranks):
	# module = nx.Graph(module)
	idx = module.nodes()

	directed_modules = []

	for connected_component in nx.connected_component_subgraphs(module):
		nodes = connected_component.nodes()
		root = nodes[ranks[nodes].argmin()]
		directed_edges = [(u, v) if ranks[u] < ranks[v] else (v, u) for u ,v in connected_component.edges() if u != v]
		directed_module = nx.DiGraph(directed_edges)

		if len(directed_module) > 0:
			directed_modules += [directed_module]

	return directed_modules

def compute_centroid(embedding,):
	centroid = embedding.sum(axis=0, keepdims=True)
	centroid = centroid / np.sqrt(centroid[:,-1:] ** 2 - np.sum(centroid[:,:-1] ** 2, axis=-1, keepdims=True))
	return centroid

def ensemble_clustering(co_association_matrix, threshold=0.5):
	clusters = []
	nodes = set(range(len(co_association_matrix)))
	added_node = False

	# remove nodes with rows all < threshold
	noise_nodes, = np.where((co_association_matrix < threshold).all(axis=1))

	nodes -= set(noise_nodes)

	while len(nodes) > 0:
		
		if not added_node:
			# add new cluster
			print ("adding new cluster")
			clusters.append([nodes.pop()])

		added_node = False
		for n in nodes:
			cluster_associativity = np.array([co_association_matrix[n, cluster].max() for cluster in clusters])
			if cluster_associativity.max() > threshold:
				clusters[cluster_associativity.argmax()].append(n)
				nodes = nodes - {n}
				added_node = True

		print (len(nodes), len(clusters))
	assignments = {n : cluster_no if len(cluster) > 1 else -1 for cluster_no, cluster in enumerate(clusters) for n in cluster} 
	assignments.update({n: -1 for n in noise_nodes})
	return assignments

def determine_core_periphery_split(cluster_ranks):
	assert len(cluster_ranks.shape) == 2 and cluster_ranks.shape[1] ==1
	kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_ranks)
	labels = k_means.labels_
	centres = kmeans.cluster_centers_
	# ensure core has label 0
	core_label = centres.argmin()
	if core_label == 1:
		labels = 1 - labels
	return labels

def assess_core_periphery_auroc(labels, ranks):
	if len(labels.shape) == 2:
		labels_ = labels[:,1]
	else:
		labels_ = labels
	return roc_auc_score(labels_, -ranks)

def compute_dispersion(cluster_embedding):
	assert len(cluster_embedding.shape) == 2
	centroid = cluster_embedding.sum(axis=0, keepdims=True)
	centroid = centroid / np.sqrt(centroid[:,-1:] ** 2  - np.sum(centroid[:,:-1] ** 2, axis=-1) )

	distances = hyperbolic_distance(cluster_embedding, centroid)
	return distances.mean()

def parse_args():
	parser = argparse.ArgumentParser(description="Density-based clustering in hyperbolic space")

	parser.add_argument("--embedding", dest="embedding_filename",  
		help="path of embedding to load.")
	parser.add_argument("--edgelist", dest="edgelist", type=str,
		help="The edgelist of the graph.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str,
		help="path to labels")
	
	parser.add_argument('--directed', action="store_true", help='flag for directed graph')

	parser.add_argument("-e", dest="max_eps", type=float, default=1.,
		help="maximum eps.")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	args = parser.parse_args()
	return args

def main():

	# 0: core
	# 1: periphery-in
	# 2: periphery-out
	colors = np.array(["r", "g", "b"])

	args = parse_args()

	embedding_filename = args.embedding_filename

	graph, features, labels, hyperboloid_embedding = load_data(args)

	poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
	ranks = 2 * np.arctanh(np.linalg.norm(poincare_embedding, 
		axis=-1, keepdims=False))

	k_core_df = pd.read_csv("edgelists/ecoli/core_numbers.csv", sep=",", index_col=0, header=None)


	print (np.corrcoef(ranks, [graph.degree(n) for n in sorted(graph.nodes())]))
	print (np.corrcoef(ranks, k_core_df[1]))


	dists = hyperbolic_distance(hyperboloid_embedding, hyperboloid_embedding)

	cores = []
	counts = np.zeros(len(graph))
	co_association_matrix = np.zeros((len(graph), len(graph)))
	max_value = np.finfo(np.float64).max

	dists_copy = dists.copy()
	dists_copy.sort(axis=1)
	dists4 = dists_copy[:,4]
	min_ = np.maximum(0.05, dists4.mean() - 2 * dists4.std())
	max_ = dists4.mean() + 2 * dists4.std()

	print (min_, max_)

	for eps in np.arange(min_, max_, 0.1):
	# for eps in np.arange(0.05, args.max_eps, 0.05):
		modules, _ = perform_clustering(dists, eps)
		num_modules = len(set(modules) - {-1})
		print ("discovered {} modules with eps = {}".format(num_modules, eps))

		fraction_of_nodes = (modules > -1).sum() / float(len(modules))
		print ("fraction of nodes in modules: {}".format(fraction_of_nodes))

		num_connected = 0
		for m in range(num_modules):
			idx = np.where(modules == m)[0]
			module = graph.subgraph(idx)
			print ("cluster =", m, "number of nodes =", len(module), 
				"number of edges =", len(module.edges()), "connected =", nx.is_connected(module.to_undirected()))

			is_connected = nx.is_connected(module.to_undirected())
			num_connected += is_connected
			
			if is_connected:
				for u in idx:
					counts[u] += 1
					for v in idx:
						# if u != v:
						co_association_matrix[u, v] += 1

				if frozenset(idx) not in cores:
					cores.append(frozenset(idx))

		print ("number of connected modules: {}".format(num_connected))

		if fraction_of_nodes == 1.:
			break

	print ("number of cores: {}".format(len(cores)))
	core_groups = {}
	is_superset = np.zeros(len(cores))
	for i in range(len(cores)):
		if is_superset[i]:
			continue
		core = cores[i]
		supersets = []
		for j in range(len(cores)):
			core_ = cores[j]
			if core < core_:
				is_superset[j] = 1
				supersets.append(j)
		core_groups.update({i: supersets})

	print ("number of core groups: {}".format(len(core_groups)))

	assignments = {n: -1 for n in sorted(graph.nodes())}
	for i in core_groups.keys():
		for n in cores[i]:
			assignments[n] = i

	co_association_matrix /= co_association_matrix.max()
	assignments_ensemble = ensemble_clustering(co_association_matrix, threshold=0.5)
		
	from sklearn.metrics import normalized_mutual_info_score
	print (normalized_mutual_info_score([assignments[n] for n in sorted(graph.nodes())],
		[assignments_ensemble[n] for n in sorted(graph.nodes())]))

	print ({c: list(assignments_ensemble.values()).count(c) for c in set(assignments_ensemble.values())})

	print (np.corrcoef(counts, k_core_df[1]))

	plt.scatter(counts, k_core_df[1])
	plt.xlabel("counts")
	plt.ylabel("kcore")
	plt.show()

	raise SystemExit

	num_clusters = len(set(assignments.values()) - {-1})
	print ("number of clusters found {}".format(num_clusters))

	print (np.array([assignments[n] for n in sorted(graph.nodes())]))
	cluster_sizes = {c: list(assignments.values()).count(c) 
		for c in set(assignments.values())}
	print (cluster_sizes)

	core_centroids = np.concatenate([compute_centroid(hyperboloid_embedding[list(cores[i])]) for i in core_groups.keys()])
	core_centroids_poincare = hyperboloid_to_poincare_ball(core_centroids)
	core_ranks = 2 * np.arctanh(np.linalg.norm(core_centroids_poincare, axis=-1, keepdims=True))

	print (core_ranks)

	print (np.corrcoef(k_shell_numbers["Kshell number"], [counts[n] for n in k_shell_numbers["GeneName"]]))

	plt.scatter(k_shell_numbers["Kshell number"], [counts[n] for n in k_shell_numbers["GeneName"]])
	plt.xlabel("k_core_number")
	plt.ylabel("core count")
	plt.show()

if __name__ == "__main__":
	main()