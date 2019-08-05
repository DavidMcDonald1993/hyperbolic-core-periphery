from __future__ import print_function

import os
import re
import glob
import argparse
import numpy as np
import networkx as nx
import pandas as pd

from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import roc_auc_score


from utils import load_data, load_embedding
from visualise import draw_graph
from tree import CoreTree

def minkowki_dot(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance(u, v):
	mink_dp = -1 - minkowki_dot(u, v)
	mink_dp = np.maximum(mink_dp, 1e-8)
	return np.arccosh(1 + mink_dp)

def perform_clustering(dists, eps):
	dbsc = DBSCAN(metric="precomputed", eps=eps, 
		n_jobs=-1, 
		min_samples=3).fit(dists)
	return dbsc.labels_, dbsc.core_sample_indices_

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def convert_module_to_directed_module(potential_core, ranks):
	# potential_core = nx.Graph(potential_core)
	idx = potential_core.nodes()

	directed_modules = []

	for connected_component in nx.connected_component_subgraphs(potential_core):
		nodes = connected_component.nodes()
		root = nodes[ranks[nodes].argmin()]
		directed_edges = [(u, v) if ranks[u] < ranks[v] else (v, u) for u, v in connected_component.edges() if u != v]
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

	graph, features, labels, hyperboloid_embedding = load_data(args)

	cores = []
	co_association_matrix = np.zeros((len(graph), len(graph)))

	dists = []

	gene_df = pd.read_csv("edgelists/ppi/gene_ids_complete.csv", sep=",", header=None, index_col=0)
	gene_df.columns = ["ORF", "GeneName"]

	# for embedding_filename in glob.iglob('embeddings/ppi/**/dim=050/*.csv', recursive=True):
	for embedding_filename in [args.embedding_filename]:

		print("loading embedding from {}".format(embedding_filename))
		hyperboloid_embedding = load_embedding(embedding_filename).reindex(sorted(graph.nodes())).values

		dists.append(hyperbolic_distance(hyperboloid_embedding, hyperboloid_embedding))

	dists = np.array(dists).max(axis=0)

	import hdbscan
	# import seaborn as sns

	def hd(x, y):
		mink_dp = x[:-1].dot(y[:-1]) - x[-1] * y[-1]
		mink_dp = -1 - mink_dp
		mink_dp = np.maximum(mink_dp, 1e-8)
		return np.arccosh(1 + mink_dp)

	clusterer = hdbscan.HDBSCAN(min_cluster_size=3, 
		gen_min_span_tree=True,
		min_samples=3,
		prediction_data=True,
		core_dist_n_jobs=-1,
		# metric="precomputed")
		algorithm="prims_kdtree",
		metric=hd)

	# clusterer.fit(dists)
	clusterer.fit(hyperboloid_embedding)

	soft_clustering = hdbscan.all_points_membership_vectors(clusterer)

	print (soft_clustering)
	raise SystemExit

	# clusterer.fit(hyperboloid_embedding)

	# clusterer.condensed_tree_.plot(select_clusters=True,
 #                               selection_palette=sns.color_palette('deep', 8))
	# plt.show()

	# print (clusterer.single_linkage_tree_.to_pandas().to_string())

	# print (clusterer.single_linkage_tree_.get_clusters(3, min_cluster_size=3))
	# raise SystemExit

	print ("one embedding ")
	num_clusters = len(set(clusterer.labels_) - {-1})
	print ("num clusters = {}".format(num_clusters))

	num_connected = 0

	for label in set(clusterer.labels_) - {-1}:
		cluster_nodes,  = np.where(clusterer.labels_ == label) 
		cluster_graph = graph.subgraph(cluster_nodes)
		core_nodes,  = np.where(clusterer.probabilities_[cluster_nodes]  == 1)
		periphery_nodes, = np.where(clusterer.probabilities_[cluster_nodes] < 1)

		is_connected = nx.is_connected(cluster_graph.to_undirected())
		num_connected += is_connected

		if is_connected:

			core_graph = graph.subgraph(core_nodes)
			periphery_graph = graph.subgraph(periphery_nodes)

			print (len(cluster_nodes), len(cluster_graph.edges()))
			print (len(core_graph), len(core_graph.edges()), nx.number_connected_components(core_graph))
			print (len(periphery_graph), len(periphery_graph.edges()), nx.number_connected_components(periphery_graph))

			print (nx.density(core_graph), 
				nx.density(periphery_graph), 
				nx.density(cluster_graph),
				nx.density(graph))

		# for n in cluster_nodes:
		# 	print (gene_df.loc[n]["ORF"])
		print ()

	print ("num_connected = {}/{}".format(num_connected, num_clusters))
	raise SystemExit
	dists = []

	for embedding_filename in glob.iglob('embeddings/ppi/**/dim=050/*.csv', recursive=True):

		print("loading embedding from {}".format(embedding_filename))
		hyperboloid_embedding = load_embedding(embedding_filename).reindex(sorted(graph.nodes())).values

		dists.append(hyperbolic_distance(hyperboloid_embedding, hyperboloid_embedding))

	dists = np.array(dists).max(axis=0)

	clusterer = hdbscan.HDBSCAN(min_cluster_size=3, 
		gen_min_span_tree=True,
		metric="precomputed")
	clusterer.fit(dists)


	print ("multple embedding ")
	num_clusters = len(set(clusterer.labels_) - {-1})
	print ("num clusters = {}".format(num_clusters))

	num_connected = 0

	for label in set(clusterer.labels_) - {-1}:
		cluster_nodes,  = np.where(clusterer.labels_ == label) 
		cluster_graph = graph.subgraph(cluster_nodes)
		# print (label, len(cluster_nodes), len(cluster_graph.edges()))
		is_connected = nx.is_connected(cluster_graph.to_undirected())
		num_connected += is_connected
		# for n in cluster_nodes:
		# 	print (gene_df.loc[n]["ORF"])
		# print ()

	print ("num_connected = {}".format(num_connected / num_clusters))

	raise SystemExit

	for embedding_filename in [args.embedding_filename]:
	# for embedding_filename in glob.iglob('embeddings/ppi/**/dim=050/*.csv', recursive=True):
		print("loading embedding from {}".format(embedding_filename))
		hyperboloid_embedding = load_embedding(embedding_filename).reindex(sorted(graph.nodes())).values

		poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
		ranks = 2 * np.arctanh(np.linalg.norm(poincare_embedding, 
			axis=-1, keepdims=False))

		# print (np.corrcoef(ranks, hyperboloid_embedding[:,-1]))
		# print (np.corrcoef(ranks, [graph.degree(n) for n in sorted(graph.nodes())]))
		# core_numbers = nx.core.core_number(graph)
		# print (np.corrcoef(ranks, [core_numbers[n] for n in sorted(graph.nodes())]))
		# print (np.corrcoef([graph.degree(n) for n in sorted(graph.nodes())], [core_numbers[n] for n in sorted(graph.nodes())]))

		dists = hyperbolic_distance(hyperboloid_embedding, hyperboloid_embedding)

		import hdbscan

		clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True,
			metric="precomputed")
		clusterer.fit(dists)

		# clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
		# 							  edge_alpha=0.6,
		# 							  node_size=80,
		# 							  edge_linewidth=2)
		# plt.show()

		# clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
		# plt.show()

		# clusterer.condensed_tree_.plot()
		# plt.show()

		print (clusterer.labels_)

		for label in set(clusterer.labels_):
			cluster_nodes,  = np.where(clusterer.labels_ == label) 
			cluster_graph = graph.subgraph(cluster_nodes)
			print (label, len(cluster_graph), len(cluster_graph.edges()))
			for n in cluster_nodes:
				assert len(nx.descendants(cluster_graph, n)) == len(cluster_nodes) - 1
			print (nx.is_connected(cluster_graph.to_undirected()))
			# out_component = set().union([*tuple(nx.descendants(graph, n)) for n in cluster_nodes])
			# print (len(out_component - set(cluster_nodes)))
			# print ()

		raise SystemExit



		# counts = np.zeros(len(graph))
		# max_value = np.finfo(np.float64).max

		dists_copy = dists.copy()
		dists_copy.sort(axis=1)
		dists3 = dists_copy[:,4]
		min_ = np.maximum(0.05, dists3.mean() - 2 * dists3.std())
		max_ = dists3.mean() + 2 * dists3.std()

		# print (min_, max_)

		# for eps in np.arange(min_, max_, 0.5):
		for eps in np.arange(0.05, args.max_eps, 0.1):
			potential_cores, _ = perform_clustering(dists, eps)
			num_potential_cores = len(set(potential_cores) - {-1})
			print ("discovered {} potential cores with eps = {}".format(num_potential_cores, eps))

			fraction_of_nodes = (potential_cores > -1).sum() / float(len(potential_cores))
			print ("fraction of nodes in cores: {}".format(fraction_of_nodes))

			num_connected = 0
			for pc in range(num_potential_cores):
				idx = np.where(potential_cores == pc)[0]
				potential_core = graph.subgraph(idx)
				print ("potential core =", pc, "number of nodes =", len(potential_core), 
					"number of edges =", len(potential_core.edges()), "connected =", nx.is_connected(potential_core.to_undirected()))

				# is_connected = all([len(nx.descendants(potential_core, u)) == len(potential_core) - 1 for u in potential_core])#nx.is_connected(potential_core.to_undirected())
				is_connected = nx.is_connected(potential_core.to_undirected())
				num_connected += is_connected
				
				if is_connected:
					for u in idx:
						# counts[u] += 1
						for v in idx:
							# if u != v:
							co_association_matrix[u, v] += 1

					if frozenset(idx) not in cores:
						cores.append(frozenset(idx))

			print ("number of connected cores: {}".format(num_connected))

			if fraction_of_nodes == 1.:
				break

	cores = sorted(cores, key=len, reverse=True)

	# for c1 in cores:
	# 	for c2 in cores:
	# 		assert c1 == c2 or c1 < c2 or c2 < c1 or len(c1.intersection(c2)) == 0, (sorted(c1), sorted(c2))

	print ("number of cores: {}".format(len(cores)))
				 
	# plt.imshow(co_association_matrix)
	# plt.show()

	core_groups = {}
	is_subset = np.zeros(len(cores))
	for i in range(len(cores)):
		if is_subset[i]:
			continue
		core = cores[i]
		subsets = []
		for j in range(len(cores)):
			core_ = cores[j]
			if core > core_:
				is_subset[j] = 1
				subsets.append(j)
		core_groups.update({i: subsets})

	for root, subsets in core_groups.items():
		for subset in subsets:
			assert cores[root] > cores[subset] 

	print ("number of core groups: {}".format(len(core_groups)))

	for root, subsets in core_groups.items():
		t = CoreTree(root, subsets, cores, depth=0)
		print (t)
		print (len(t))
		print ()
	raise SystemExit

	assignments = {n: -1 for n in sorted(graph.nodes())}
	for i in core_groups.keys():
		print (cores[i])
		print (cores[core_groups[i][0]])
		print ()
		for n in cores[i]:
			assignments[n] = i

	raise SystemExit

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