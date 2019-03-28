import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

def load_embedding(filename):
	assert filename.endswith(".csv")
	embedding_df = pd.read_csv(filename, index_col=0)
	return embedding_df


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