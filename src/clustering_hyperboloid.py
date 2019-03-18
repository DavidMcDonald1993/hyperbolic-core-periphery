from __future__ import print_function

import os
import re
import argparse
import numpy as np
import networkx as nx
import pandas as pd

from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt


from sklearn.cluster import DBSCAN

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

def load_embedding(filename):
	assert filename.endswith(".csv")
	embedding_df = pd.read_csv(filename, index_col=0)
	return embedding_df


def perform_clustering(dists, eps):
	dbsc = DBSCAN(metric="precomputed", eps=eps, n_jobs=-1, min_samples=3)
	labels = dbsc.fit_predict(dists)
	return labels

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

def grow_forest(data_train, directed_modules, ranks, feature_names, bootstrap=True, ):

	n = data_train.shape[0]


	forest = []
	all_oob_samples = []

	for directed_module in directed_modules:
		feats = directed_module.nodes()
		root = feats[ranks[feats].argmin()]
		assert nx.is_connected(directed_module.to_undirected())

		if bootstrap:
			idx = np.random.choice(n, size=n, replace=True)
			_data_train = data_train[idx]
		else:
			_data_train = data_train

		tree = TopologyConstrainedTree(parent_index=None, index=root, g=directed_module, 
			data=_data_train, feature_names=feature_names, depth=0, max_depth=np.inf, min_samples_split=2, min_neighbours=1)

		if bootstrap:
			oob_samples = list(set(range(n)) - set(idx))
			oob_samples = data_train[oob_samples]
			all_oob_samples.append(oob_samples)

			# oob_prediction = tree.predict(oob_samples)
			# oob_prediction_accuracy = tree.prediction_accuracy(oob_samples[:,-1], oob_prediction)
			# print (n, set(idx), len(oob_samples), oob_prediction_accuracy)

		forest.append(tree)
	return forest, all_oob_samples

def evaluate_modules_on_test_data(features, labels, directed_modules, ranks, feature_names,
	n_repeats=10, test_size=0.3, ):

	data = np.column_stack([features, labels])

	f1_micros = []

	sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)
	for split_train, split_test in sss.split(features, labels):

		data_train = data[split_train]
		data_test = data[split_test]

		forest, _ = grow_forest(data_train, directed_modules, ranks, feature_names)

		test_prediction = np.array([t.predict(data_test) for t in forest])
		test_prediction = test_prediction.mean(axis=0) > 0.5
		f1_micro = f1_score(data_test[:,-1], test_prediction, average="micro")
		f1_micros.append(f1_micro)

	return np.mean(f1_micros)

def determine_feature_importances(forest, all_oob_samples):

	n_trees = len(forest)
	n_features = all_oob_samples[0].shape[1] - 1
	feature_importances = np.zeros((n_trees, n_features),)
	feature_pair_importances = np.zeros((n_trees, n_features, n_features), )

	for i, tree, oob_samples in zip(range(n_trees), forest, all_oob_samples):
		oob_sample_prediction = tree.predict(oob_samples)
		oob_sample_accuracy = tree.prediction_accuracy(oob_samples[:,-1], oob_sample_prediction)

		for feature in range(n_features):
			_oob_samples = oob_samples.copy()
			np.random.shuffle(_oob_samples[:,feature])
			permuted_prediction = tree.predict(_oob_samples)
			permuted_prediction_accuracy = tree.prediction_accuracy(_oob_samples[:,-1], permuted_prediction)
			feature_importances[i, feature] = oob_sample_accuracy - permuted_prediction_accuracy

		# for f1 in range(n_features):
		# 	for f2 in range(n_features):
		# 		_oob_samples = oob_samples.copy()
		# 		np.random.shuffle(_oob_samples[:,f1])
		# 		np.random.shuffle(_oob_samples[:,f2])

		# 		permuted_prediction = tree.predict(_oob_samples)
		# 		permuted_prediction_accuracy = tree.prediction_accuracy(_oob_samples[:,-1], permuted_prediction)
		# 		feature_importances[i, feature] = oob_sample_accuracy - permuted_prediction_accuracy

	return feature_importances.mean(axis=0), feature_pair_importances.mean(axis=0)


def plot_disk_embeddings(edges, poincare_embedding, modules,):

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges) 

	all_modules = sorted(set(modules))
	num_modules = len(all_modules)
	colors = np.random.rand(num_modules, 3)

	fig = plt.figure(figsize=[14, 7])
	
	ax = fig.add_subplot(111)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	u_emb = poincare_embedding[edges[:,0]]
	v_emb = poincare_embedding[edges[:,1]]
	plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)
	for i, m in enumerate(all_modules):
		idx = modules == m
		plt.scatter(poincare_embedding[idx,0], poincare_embedding[idx,1], s=10, 
			c=colors[i], label="module={}".format(m) if m > -1 else "noise", zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=4)

	# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=3)
	plt.show()
	# plt.savefig(path)
	plt.close()

def load_data(args):

	edgelist_filename = args.edgelist
	features_filename = args.features
	labels_filename = args.labels

	assert not edgelist_filename == "none", "you must specify and edgelist file"

	graph = nx.read_edgelist(edgelist_filename, delimiter="\t", nodetype=int,
		create_using=nx.DiGraph() if args.directed else nx.Graph())

	if not features_filename == "none":

		if features_filename.endswith(".csv"):
			features = pd.read_csv(features_filename, index_col=0, sep=",")
			features = features.reindex(graph.nodes()).values
			features = StandardScaler().fit_transform(features)
		else:
			raise Exception

	else: 
		features = None

	if not labels_filename == "none":

		if labels_filename.endswith(".csv"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(graph.nodes()).values.flatten()
			assert len(labels.shape) == 1
		else:
			raise Exception

	else:
		labels = None

	embedding_filename = args.embedding_filename
	embedding_df = load_embedding(embedding_filename)
	embedding = embedding_df.reindex(graph.nodes()).values

	graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")

	return graph, features, labels, embedding

def parse_args():
	parser = argparse.ArgumentParser(description="Density-based clustering in hyperbolic space")

	parser.add_argument("--embedding", dest="embedding_filename",  
		help="path of embedding to load.")
	parser.add_argument("--edgelist", dest="edgelist", type=str,
		help="The edgelist of the graph.")
	parser.add_argument("--features", dest="features", type=str, default="none",
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str,
		help="path to labels")
	
	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')


	parser.add_argument("-e", dest="max_eps", type=float, default=1,
		help="maximum eps.")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	embedding_filename = args.embedding_filename

	graph, features, labels, hyperboloid_embedding = load_data(args)

	poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
	ranks = np.sqrt(np.sum(np.square(poincare_embedding), axis=-1, keepdims=False))
	assert (ranks<1).all()
	assert (ranks.argsort() == hyperboloid_embedding[:,-1].argsort()).all()

	dists = hyperbolic_distance(hyperboloid_embedding, hyperboloid_embedding)

	plt.imshow(dists)
	plt.show()
	raise SystemExit

	best_eps = -1
	best_f1 = 0
	heatmap = np.zeros((len(graph), len(graph)), )
	for eps in np.arange(0.1, args.max_eps, 0.1):
		modules = perform_clustering(dists, eps)
		num_modules = len(set(modules) - {-1})
		print ("discovered {} modules with eps = {}".format(num_modules, eps))

		fraction_of_nodes = (modules > -1).sum() / float(len(modules))
		print ("fraction of nodes in modules: {}".format(fraction_of_nodes))

		num_connected = 0
		# directed_modules = []
		for m in range(num_modules):
			idx = np.where(modules == m)[0]
			for u in idx:
				for v in idx:
					heatmap[u,v] += 1
			module = graph.subgraph(idx)
			print ("module =", m, "number of nodes =", len(module), 
				"number of edges =", len(module.edges()))
			num_connected += nx.is_connected(module)
			# directed_modules += convert_module_to_directed_module(module, ranks)

		print ("number of connected modules: {}".format(num_connected))

		if fraction_of_nodes == 1.:
			break

	plt.imshow(heatmap)
	
	draw_graph(graph.edges(), poincare_embedding, labels=perform_clustering(dists, eps=0.5), path=None)

if __name__ == "__main__":
	main()