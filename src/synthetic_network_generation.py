import sys
if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")

import numpy as np
import networkx as nx

# import cpalgorithm as cp
# import json
import os
import pickle as pkl

def build_stochastic_block_model(num_nodes, core_periphery_probs, connection_probs, 
								 self_loops=False, directed=False, seed=0):
	
	np.random.seed(seed)
	
	assert len(core_periphery_probs) * 2 == len(connection_probs)
	if not core_periphery_probs.sum() == 1:
		core_periphery_probs /= core_periphery_probs.sum()
	
	random_nodes = np.random.permutation(num_nodes)
	
	nodes = {}
	
	core_number = 0
	
	for core_p, perip_p in core_periphery_probs:
		# print (core_p, perip_p)
		num_core = int( np.ceil(num_nodes*core_p, ))
		num_periphery = int(np.ceil(num_nodes*perip_p, ))
		# print (num_core, num_periphery)
		nodes.update({ n : (core_number, 1)  for n in random_nodes[:num_core]})

		random_nodes = random_nodes[num_core:]
		
		nodes.update({ n : (core_number, 0) for n in random_nodes[:num_periphery]})
		
		random_nodes = random_nodes[num_periphery:]

		core_number += 1

	assert len(nodes) == num_nodes, (len(nodes), num_nodes)
		
	node_pairs = ((n1, n2) for n1 in range(num_nodes) 
				  for n2 in (range(n1+1, num_nodes) if not directed and not self_loops
					else range(n1, num_nodes) if not directed 
					else range(num_nodes)) )
#     node_pairs = ([(n1, n2) for n1 in range(num_nodes) for n2 in range(num_nodes)] if directed and self_loops
#               else np.triu_indices(num_nodes, k=1-self_loops))
	
	probs = [connection_probs[2*nodes[n1][0] + 1-nodes[n1][1], 
							2*nodes[n2][0] + 1-nodes[n2][1]] for (n1, n2) in node_pairs]
	
	adj_flat = np.random.rand(len(probs)) < probs
	
	if directed and self_loops:
		adj = adj_flat.reshape(num_nodes, num_nodes)
	else:
		adj = np.zeros((num_nodes, num_nodes))
		idx = np.triu_indices(num_nodes, k=1-self_loops)
		adj[idx] = adj_flat
		adj[idx[::-1]] = adj_flat
 
	return nodes, adj

def main():

	data_dir = "edgelists/"

	num_nodes = 400
	num_seeds = 30

	exps = ["one_core", "two_core", "two_core_with_residual"]

	for exp in exps:

		if exp == "one_core":
			pi1 = 1./4
			pi2 = 3./4
			core_periphery_probs = np.array([[pi1, pi2]])
		elif exp == "two_core":
			pi1 = 1./8
			pi2 = 3./8
			core_periphery_probs = np.array([[pi1, pi2],
											[pi1, pi2]])
		elif exp == "two_core_with_residual":
			pi1 = 1./9
			pi2 = 1./3
			core_periphery_probs = np.array([[pi1, pi2],
											[pi1, pi2],
											[pi1, 0]])
		else: 
			raise Exception

		for theta1 in np.arange(0.10, 1.0, 0.05):
			for theta2 in np.arange(0.05, theta1, 0.05):

				if exp == "one_core":
					connection_probs = np.array([[theta1, theta1],
												[theta1, theta2]])
				elif exp == "two_core":
					connection_probs = np.array([[theta1, theta1, theta2, theta2],
												[theta1, theta2, theta2, theta2],
												[theta2, theta2, theta1, theta1],
												[theta2, theta2, theta1, theta2]])
				elif exp == "two_core_with_residual":
					connection_probs = np.array([[theta1, theta1, theta2, theta2, theta2, 0],
												[theta1, theta2, theta2, theta2, theta2, 0],
												[theta2, theta2, theta1, theta1, theta2, 0],
												[theta2, theta2, theta1, theta2, theta2, 0],
												[theta2, theta2, theta2, theta2, theta2, 0],
												[0,0,0,0,0,0]])


				for seed in range(num_seeds):

						directory = os.path.join(data_dir, "synthetic", exp)
						if not os.path.exists(directory):
							print ("Making directory: {}".format(directory))
							os.makedirs(directory, exist_ok=True)

						filename = "theta1={:.02f}-theta2={:.02f}-seed={:02d}".format(theta1, theta2, seed)
						edgelist_filename = os.path.join(directory, filename + ".edgelist")
						node_label_filename = os.path.join(directory, filename + ".pkl")

						if os.path.exists(edgelist_filename):
							print ("{} already exists".format(filename))
							continue


						nodes, adj = build_stochastic_block_model(num_nodes, 
							core_periphery_probs, 
							connection_probs,
							seed=seed)

						g = nx.from_numpy_matrix(adj)
						nx.write_edgelist(g, edgelist_filename, delimiter="\t")
						with open(node_label_filename, "wb") as f:
							pkl.dump(nodes, f, pkl.HIGHEST_PROTOCOL)

						print ("Completed {}".format(filename))





if __name__ == "__main__":
	main()
