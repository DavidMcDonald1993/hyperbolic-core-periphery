from __future__ import print_function

import os
import re
import glob
import argparse
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing.pool import Pool

import functools
import itertools

import hdbscan

from utils import load_data, load_embedding

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

def exemplars(cluster_id, condensed_tree):
	raw_tree = condensed_tree._raw_tree
	# Just the cluster elements of the tree, excluding singleton points
	cluster_tree = raw_tree[raw_tree['child_size'] > 1]
	# Get the leaf cluster nodes under the cluster we are considering
	leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
	# Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
	result = np.array([])
	for leaf in leaves:
		max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
		points = raw_tree['child'][(raw_tree['parent'] == leaf) &
								   (raw_tree['lambda_val'] == max_lambda)]
		result = np.hstack((result, points))
	return result.astype(np.int)

def min_dist_to_exemplar(data, cluster_exemplars, dist=hyperbolic_distance):
	dists = dist(data, data[cluster_exemplars.astype(np.int32)])
	return dists.min(axis=-1)

def dist_vector(data, exemplar_dict, ):
	result = {}
	for cluster in exemplar_dict:
		result[cluster] = min_dist_to_exemplar(data, exemplar_dict[cluster],)
	return np.array(list(result.values())).T

def dist_membership_vector(data, exemplar_dict, softmax=False):
	if softmax:
		result = np.exp(1./dist_vector(data, exemplar_dict, ))
		result[~np.isfinite(result)] = np.finfo(np.double).max
	else:
		result = 1./dist_vector(data, exemplar_dict, )
		result[~np.isfinite(result)] = np.finfo(np.double).max
	result /= result.sum(axis=-1, keepdims=True)
	return result

def max_lambda_val(cluster, tree):
	cluster_tree = tree[tree['child_size'] > 1]
	leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster)
	max_lambda = 0.0
	for leaf in leaves:
		max_lambda = max(max_lambda,
						 tree['lambda_val'][tree['parent'] == leaf].max())
	return max_lambda

def points_in_cluster(cluster, tree):
	leaves = hdbscan.plots._recurse_leaf_dfs(tree, cluster)
	return leaves

def merge_height(point_cluster, tree, point_dict):
	point, cluster = point_cluster
	cluster_row = tree[tree['child'] == cluster]
	# cluster_height = cluster_row['lambda_val'][0]
	if point in point_dict[cluster]:
		merge_row = tree[tree['child'] == float(point)][0]
		return point_cluster, merge_row['lambda_val']
	else:
		while point not in point_dict[cluster]:
			parent_row = tree[tree['child'] == cluster]
			cluster = parent_row['parent'].astype(np.float64)[0]
		for row in tree[tree['parent'] == cluster]:
			child_cluster = float(row['child'])
			if child_cluster == point:
				return point_cluster, row['lambda_val']
			if child_cluster in point_dict and point in point_dict[child_cluster]:
				return point_cluster, row['lambda_val']

def per_cluster_scores(merge_heights, tree, max_lambda_dict, ):
	point_rows = (tree[tree['child'] == point] for point in range(len(merge_heights)))
	point_clusters = (float(point_row[0]['parent']) for point_row in point_rows)
	max_lambdas = np.array([[max_lambda_dict[point_cluster] + 1e-8]
		for point_cluster in point_clusters]) # avoid zero lambda vals in odd cases

	results = max_lambdas / (max_lambdas - merge_heights)

	# result = {}
 #    point_row = tree[tree['child'] == point]
 #    point_cluster = float(point_row[0]['parent'])
 #    max_lambda = max_lambda_dict[point_cluster] + 1e-8 # avoid zero lambda vals in odd cases

 #    for c in cluster_ids:
 #        height = merge_height(point, c, tree, point_dict)
 #        result[c] = (max_lambda / (max_lambda - height))

	return results

def outlier_membership_vector(merge_heights, tree,
							  max_lambda_dict, softmax=False):

	result = per_cluster_scores(merge_heights, tree._raw_tree, max_lambda_dict)
	# result = np.array([list(per_cluster_scores(point,
	# 						cluster_ids,
	# 						tree,
	# 						max_lambda_dict,
	# 						point_dict
	# 					).values()) for point in range(len(data))])
	if softmax:
		result -= result.max(axis=-1, keepdims=True)

		result = np.exp(result)

		result[~np.isfinite(result)] = np.finfo(np.double).max
		
	result /= result.sum(axis=-1, keepdims=True)
	return result

def combined_membership_vector(data, tree, exemplar_dict, merge_heights,
							   max_lambda_dict, softmax=False):
	# raw_tree = tree._raw_tree
	dist_vec = dist_membership_vector(data, exemplar_dict, softmax)
	outl_vec = outlier_membership_vector(merge_heights, tree,
										  max_lambda_dict, softmax)
	result = dist_vec * outl_vec
	result /= result.sum(axis=-1, keepdims=True)
	return result

def prob_in_some_cluster(merge_heights, cluster_ids, max_lambda_dict):
	# heights = []
	# for cluster in cluster_ids:
	# 	heights.append(merge_height(point, cluster, tree._raw_tree, point_dict))
	# heights = np.array([[merge_height(point, cluster, tree._raw_tree, point_dict)
	# 	for cluster in cluster_ids]
	# 	for point in range(len(data))])
	# height = heights.max(axis=-1)
	# nearest_cluster = cluster_ids[np.argmax(heights)]
	max_lambda = np.array([[max_lambda_dict[cluster_ids[i]]] 
		for i in merge_heights.argmax(axis=-1)])
	assert len(max_lambda.shape) == 2
	return merge_heights.max(axis=-1, keepdims=True) / max_lambda

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def compute_centroid(embedding,):
	centroid = embedding.sum(axis=0, keepdims=True)
	centroid = centroid / np.sqrt(centroid[:,-1:] ** 2 - np.sum(centroid[:,:-1] ** 2, axis=-1, keepdims=True))
	return centroid

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

	poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
	ranks = 2 * np.arctanh(np.linalg.norm(poincare_embedding, 
			axis=-1, keepdims=False))


	# gene_df = pd.read_csv("edgelists/ppi/gene_ids_complete.csv", sep=",", header=None, index_col=0)
	# gene_df.columns = ["ORF", "GeneName"]

	# with open("string_all_genes.txt", "w") as f:
	# 	for gene in gene_df["ORF"]:
	# 		f.write("{}\n".format(gene))

	# raise SystemExit

	# k_shell_df = pd.read_csv("edgelists/grn/k_shell_numbers.csv", sep=",", index_col=0)

	# genes_of_interest = k_shell_df["GeneName"]

	# print (np.corrcoef(ranks[genes_of_interest], k_shell_df["Kshell number"])) 
	# plt.scatter(ranks[genes_of_interest], k_shell_df["Kshell number"])
	# plt.xlabel("distance from origin")
	# plt.ylabel("k shell number")
	# plt.show()

	core_numbers = nx.algorithms.core.core_number(graph)
	print (np.corrcoef(ranks, [core_numbers[n] for n in sorted(graph.nodes())]))
	plt.scatter(ranks, [core_numbers[n] for n in sorted(graph.nodes())])
	plt.xlabel("distance from origin")
	plt.ylabel("k core number")
	plt.show()

	dists = []

	# for embedding_filename in glob.iglob('embeddings/ppi/**/dim=050/*.csv', recursive=True):
	# for embedding_filename in glob.iglob('embeddings/ppi/**/*.csv', recursive=True):
	for embedding_filename in [args.embedding_filename]:

		print("loading embedding from {}".format(embedding_filename))
		hyperboloid_embedding = load_embedding(embedding_filename).reindex(sorted(graph.nodes())).values

		dists.append(hyperbolic_distance(hyperboloid_embedding, hyperboloid_embedding))

	dists = np.array(dists).max(axis=0)

	clusterer = hdbscan.HDBSCAN(min_cluster_size=5, 
		gen_min_span_tree=True,
		# min_samples=3,
		core_dist_n_jobs=-1,
		allow_single_cluster=False,
		metric="precomputed")

	clusterer.fit(dists)

	tree = clusterer.condensed_tree_

	tree.plot(select_clusters=True, )
	plt.show()

	assignments = clusterer.labels_
	probabilities = clusterer.probabilities_

	assigned_nodes, = np.where(assignments >= 0, )
	noise_nodes, = np.where(assignments == -1, )

	assert not (assignments == -1).all()
	# print ((assignments >= 0).sum() / len(assignments))

	# print (np.corrcoef(probabilities[assigned_nodes], [core_numbers[n] for n in assigned_nodes]))
	# plt.scatter(probabilities[assigned_nodes], [core_numbers[n] for n in assigned_nodes])
	# plt.xlabel("probabilities")
	# plt.ylabel("k shell number")
	# plt.show()

	# gene_df = pd.read_csv("edgelists/ppi/gene_ids_complete.csv", sep=",", header=None, index_col=0)
	# gene_df.columns = ["ORF", "GeneName"]

	# for c in set(assignments) - {-1}:
	# 	core_nodes, = np.where(assignments == c,)
	# 	with open("{:03d}.genes".format(c), mode="w") as f:
	# 		for core_node in core_nodes:
	# 			f.write("{}\n".format(gene_df.loc[core_node]["ORF"]))


	# raise SystemExit

	N = len(graph)

	cluster_ids = tree._select_clusters()
	print ("SELECTED CLUSTERS", cluster_ids)
	exemplar_dict = {c: exemplars(c, tree) for c in cluster_ids}
	raw_tree = tree._raw_tree
	all_possible_clusters = np.arange(N, raw_tree['parent'].max() + 1).astype(np.float64)
	max_lambda_dict = {c: max_lambda_val(c, raw_tree) for c in all_possible_clusters}
	point_dict = {c: set(points_in_cluster(c, raw_tree)) for c in all_possible_clusters}

	print ("COMPUTING merge_heights")
	with Pool(processes=None) as p:
		merge_heights = p.map(functools.partial(merge_height, 
			tree=raw_tree, 
			point_dict=point_dict), 
			itertools.product(range(N), cluster_ids))
	assert len(merge_heights) == N * len(cluster_ids)
	merge_heights_df = pd.DataFrame(0, index=range(N), columns=cluster_ids, )
	for (point, cluster), merge_height_ in merge_heights:
		assert merge_height_ != 0
		assert merge_height((point, cluster), raw_tree, point_dict)[1]  == merge_height_
		assert point in merge_heights_df.index, point
		assert cluster in merge_heights_df.columns, cluster
		merge_heights_df.loc[point, cluster] = merge_height_
	merge_heights = merge_heights_df.values
	# merge_heights = np.array([[merge_height(point, cluster, raw_tree, point_dict)
	# 	for cluster in cluster_ids]
	# 	for point in range(N)])

	print ("DONE")
	print ("COMPUTING combined_membership_vectors")
	combined_membership_vectors = combined_membership_vector(hyperboloid_embedding, 
		tree, exemplar_dict, merge_heights, max_lambda_dict, softmax=True)
	# combined_membership_vectors = combined_membership_vector(hyperboloid_embedding, tree, exemplar_dict, cluster_ids,
	# 						   max_lambda_dict, point_dict, softmax=False)
	print ("COMPUTING prob_in_some_cluster")
	probs_in_clusters = prob_in_some_cluster(merge_heights, cluster_ids, max_lambda_dict)

	print ("DONE")

	soft_probabilities = combined_membership_vectors * probs_in_clusters


	if False:
		# labels = np.array([g in k_shell_df["GeneName"] for g in range(N)], dtype=int)
		# from sklearn.metrics import roc_auc_score
		# print (roc_auc_score(labels, soft_probabilities[:,0]))
		# print (roc_auc_score(labels, probabilities))
		print (np.corrcoef(soft_probabilities[genes_of_interest,0], 
			# [core_numbers[n] for n in sorted(graph.nodes())]))
			k_shell_df["Kshell number"]))
		plt.scatter(soft_probabilities[genes_of_interest,0], 
			# [core_numbers[n] for n in sorted(graph.nodes())])
			k_shell_df["Kshell number"])
		plt.xlabel("soft probabilities")
		plt.ylabel("k shell number")
		plt.show()


		raise SystemExit

	# print (np.corrcoef(soft_probabilities[:,0], [core_numbers[n] for n in sorted(graph.nodes())]))
	# plt.scatter(soft_probabilities[:,0], [core_numbers[n] for n in sorted(graph.nodes())])
	# plt.xlabel("probabilities")
	# plt.ylabel("k shell number")
	# plt.show()
	# raise SystemExit

	# print (assignments[assigned_nodes][:100])
	# print (soft_probabilities[assigned_nodes].argmax(axis=-1)[:100])
	# # assert np.allclose(assignments[assigned_nodes], soft_probabilities[assigned_nodes].argmax(axis=-1))
	# idx, = np.where(assignments[assigned_nodes] != soft_probabilities[assigned_nodes].argmax(-1))
	# for i in idx:
	# 	print (assignments[assigned_nodes][i])
	# 	print (soft_probabilities[assigned_nodes][i].argsort()[::-1])
	# 	print ()



	print ("num assigned_nodes =", len(assigned_nodes), "num noise_nodes =", len(noise_nodes))
	for c in sorted(set(assignments) - {-1}):
		core_nodes, = np.where(assignments == c)
		periphery_nodes, = np.where(soft_probabilities[noise_nodes].argmax(-1) == c)
		print (c, len(core_nodes), len(periphery_nodes))
		print (nx.is_connected(graph.subgraph(core_nodes).to_undirected()))
		print (nx.is_connected(graph.subgraph(np.append(core_nodes, periphery_nodes)).to_undirected()))
		print (soft_probabilities[periphery_nodes,c].max(), soft_probabilities[periphery_nodes,c].min())
		core_centroid = compute_centroid(hyperboloid_embedding[core_nodes])
		print ("CORE CENTROID", core_centroid)
		core_centroid_poincare = hyperboloid_to_poincare_ball(core_centroid)
		print ("CORE CENTROID POINCARE", core_centroid_poincare)
		r = 2 * np.arctanh(np.linalg.norm(core_centroid_poincare, axis=-1, keepdims=True))
		print ("r =", r)
		idx = soft_probabilities[core_nodes, c].argsort()[::-1]
		# for nc in core_nodes[idx]:
		# 	print (nc, soft_probabilities[nc, c])
		# 	if nc in k_shell_df["GeneName"]:
		# 		print (k_shell_df["Kshell number"][k_shell_df["GeneName"] == nc])
		# 	else:
		# 		print (nc, "not in df")

		# with open("edgelists/ppi/{}_{}.core".format(c, r), "w") as f:
		# 	for n in core_nodes:
		# 		f.write("{}\n".format(gene_df.loc[n]["GeneName"]))



if __name__ == "__main__":
	main()