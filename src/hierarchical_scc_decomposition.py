import os
import argparse

import numpy as np 
import networkx as nx 
import pandas as pd 
import itertools

from networkx.drawing.nx_agraph import write_dot, graphviz_layout, to_agraph

import matplotlib.pyplot as plt

def find_cycles(u, n, g, start, l=set()):   
    
    if n==0:
        assert u == start
        return [[u]]
    
    l_ = l .union( {u} )
    if n > 1:
        neighbors = set(g.neighbors(u)) - l_
    else:
        neighbors = set(g.neighbors(u)) . intersection({start})
        
    paths = ( [u] + cycle
    for neighbor in neighbors
    for cycle in find_cycles(neighbor, n-1, g, start, l_) )
    return paths

def score_subgraph(g, groups):
    # nodes = set(g)

    subgraph = g.subgraph(groups)
    k_in = len([(u, v) for u, v in subgraph.edges() 
        if u != v])
    # k_all = nx.number_of_edges(subgraph)

    return  k_in / \
        sum((len(list(g.neighbors(u))) for u in subgraph))
        # (k_all + nx.cut_size(g, groups, nodes-groups)) ** 1.
    # return np.mean([nx.cut_size(g.subgraph([n1] + [n2]), 
    #     [n1], [n2]) / \
    #     max(nx.number_of_edges(g.subgraph([n1] + [n2])), 1e-7)
    #     for n1, n2 in itertools.combinations(groups, 2)]) 

def bottom_up_partition(g, subgraph_sizes=[2, 3]):

    g = g.copy()
    g = nx.relabel_nodes(g, mapping=lambda x: frozenset([x]))
    g = nx.MultiDiGraph(g)
    h = nx.DiGraph()
    h.add_nodes_from( g.nodes() )

    subgraph_scores = {}

    i = 0
    for s in subgraph_sizes:
        for n in g.nodes():
            for cycle in map(frozenset, find_cycles(n, s, 
                g, start=n)):
                if cycle in subgraph_scores:
                    continue
                # assert nx.is_strongly_connected(g.subgraph(cycle))
                subgraph_scores[cycle] = \
                    score_subgraph(g, cycle)
            print ("done", i, "/", len(g) * len(subgraph_sizes))
            i += 1
            
    while len(g) > 1:

        print ("number of nodes in g:", len(g))

        # chosen_subgraph = max(subgraph_scores, 
        #     key=lambda x: (subgraph_scores[x], len(x)))
        
        # print ("merging", unpack(chosen_subgraph), 
        #     "with score", subgraph_scores[chosen_subgraph])
        
        sorted_subgraphs = sorted(subgraph_scores, 
            key=lambda x: subgraph_scores[x],#(subgraph_scores[x], len(x)),
            reverse=True)
        chosen_subgraph = sorted_subgraphs.pop(0)
        chosen_subgraph_score = subgraph_scores[chosen_subgraph]
        # chosen_subgraph_length = len(chosen_subgraph)
        chosen_subgraphs = [chosen_subgraph]

        # assert chosen_subgraph[0] == chosen_subgraph_
        for subgraph in sorted_subgraphs:
            if (False#len(subgraph) < chosen_subgraph_length
                or subgraph_scores[subgraph] < chosen_subgraph_score):
                break
            chosen_subgraphs.append(subgraph)

        if len(chosen_subgraphs) > 1:

            # chosen_subgraphs = chosen_subgraphs[:1]

            overlaps = np.array([[len(x.intersection(y)) 
                for y in chosen_subgraphs]
                for x in chosen_subgraphs])
            overlap_g = nx.Graph(overlaps)
            chosen_subgraphs = [frozenset().union([x 
                for c in cc for x in chosen_subgraphs[c]]) 
                for cc in nx.connected_components(overlap_g)]

        for chosen_subgraph in chosen_subgraphs:

            # remove scores associated with merged nodes
            subgraph_scores = {k: v 
                for k, v in subgraph_scores.items()
                if not any([x in k for x in chosen_subgraph])}
            
            # merge subgraph into meta-node
            g.add_node(chosen_subgraph)
            for n in chosen_subgraph:
                for u, _ in g.in_edges(n):
                    # if u == chosen_subgraph:
                        # continue
                    g.add_edge(u, chosen_subgraph)
                for _, v in g.out_edges(n):
                    # if v == chosen_subgraph:
                        # continue
                    g.add_edge(chosen_subgraph, v)
                g.remove_node(n)

            # add chosen subgraph to h
            h.add_node(chosen_subgraph)
            for n in chosen_subgraph:
                h.add_edge(n, chosen_subgraph)

            # add cycles containing new node
            for s in subgraph_sizes:
                for cycle in map(frozenset, 
                find_cycles(chosen_subgraph, s, 
                g, start=chosen_subgraph)):
                    # assert nx.is_strongly_connected(g.subgraph(cycle))
                    if cycle in subgraph_scores:
                        continue
                    subgraph_scores[cycle] = \
                        score_subgraph(g, cycle)

    return h


def partition_all_sccs(g, subgraph_sizes=[2, 3]):
    h = nx.DiGraph()
    roots = []
    for scc in nx.strongly_connected_component_subgraphs(g):
        scc_tree = bottom_up_partition(scc, 
            subgraph_sizes=subgraph_sizes)
        degrees = dict(scc_tree.out_degree())
        root = min(degrees, key=degrees.get)
        roots.append(root)
        h = nx.union(h, scc_tree)

    # add final root to represent whole network
    all_nodes = frozenset(g.nodes())
    if len(roots) > 1:
        for root in roots:
            h.add_edge(root, all_nodes)

    return h

def unpack(x):
    if not any([isinstance(x_, frozenset) for x_ in x]):
        return list(x)
    else:
        return [_x for x_ in x for _x in unpack(x_)]

def main():

    datasets = ["grn_small"]

    # datasets = [
    #     "yeast_cell_cycle", 
    #     "mammalian_cortical_development",
    #     "arabidopsis_thaliana_development",
    #     "mouse_myeloid_development"
    # ]

    all_control_nodes = [
        ["Cln3", "Clb5_6", "Clb1_2"],
        ["Fgf8_g", "Emx2_g", "Sp8_g", "Fgf8_p", "Emx2_p", "Sp8_p"],
        ["AP1"],
        ["GATA-1"]
    ]

    for dataset, control_nodes in zip(datasets, all_control_nodes):
    
        print ("decomposing", dataset)

        g = nx.read_weighted_edgelist("datasets/{}/edgelist.tsv".format(dataset), 
            nodetype=int, 
            create_using=nx.DiGraph())
        mapping = pd.read_csv("datasets/{}/gene_ids.csv".format(dataset), 
                          index_col=0, header=None)[1].to_dict()
        g = nx.relabel_nodes(g, mapping=mapping)  

        zero_weight_edges = [(u, v) 
            for u, v, w in g.edges(data="weight") if w == 0]
        g.remove_edges_from(zero_weight_edges)

        # for n in control_nodes:
        #     assert n in g, n

        # nx.set_node_attributes(g, name="color", 
        #     values={n: ("red" if n in control_nodes else "black") 
        #     for n in g})

        h = partition_all_sccs(g, subgraph_sizes=[2,3,] )

        if not os.path.exists("images/{}".format(dataset)):
            os.mkdir("images/{}".format(dataset))

        h = h.reverse()

        # map_ = {}
        # for i, n in enumerate(h.nodes):
        #     g_ = nx.MultiDiGraph(g.subgraph(unpack(n)))
            
        #     map_.update({n: i})
        #     for no, child in enumerate(h.neighbors(n)):
        #         # make metanode
        #         node = "metanode_{}".format(no)
        #         g_.add_node(node, label="", 
        #         image="images/{}/subgraph_{}.png".format(\
        #             dataset, map_[child]))
        #         for n_ in unpack(child):
        #             for u, _ in g_.in_edges(n_):
        #                 if u == node:
        #                     continue
        #                 g_.add_edge(u, node)
        #             for _, v in g_.out_edges(n_):
        #                 if v == node:
        #                     continue
        #                 g_.add_edge(node, v)
        #             g_.remove_node(n_)
                    
        #     plot_filename = "images/{}/subgraph_{}.png".format(\
        #         dataset, i)
        #     g_.graph['edge'] = {'arrowsize': '.8', 'splines': 'curved'}
        #     g_.graph['graph'] = {'scale': '3'}

        #     a = to_agraph(g_)
        #     a.layout('dot')   
        #     a.draw(plot_filename)

        #     print ("plotted", plot_filename)
        
        h = h.reverse()

        # nx.set_node_attributes(h, name="image", 
        #    values={n: "images/{}/subgraph_{}.png".format(dataset,i) 
        #    for i, n in enumerate(h.nodes())})
        nx.set_node_attributes(h, name="label", #values="")
            values={n: (len(unpack(n)) 
            if len(unpack(n)) > 1 else unpack(n) )
            for n in h})

        tree_plot_filename = "images/{}/scc_tree.png".format(dataset)
        h.graph['edge'] = {'arrowsize': '.8', 'splines': 'curved'}
        h.graph['graph'] = {'scale': '3'}

        a = to_agraph(h)
        a.layout('dot')   
        a.draw(tree_plot_filename)

        print ("plotted", tree_plot_filename)

    # k_shell_numbers = pd.read_csv("datasets/grn_small/k_shell_numbers.csv", index_col=1)

    # scc = max(nx.strongly_connected_component_subgraphs(g), key=len)
    # in_scc = np.array([gene in scc 
    # for gene in map(lambda x: int(x), k_shell_numbers.index)])

    # out_degrees = dict(h.out_degree())
    # root = min(out_degrees, key=out_degrees.get)

    # sps = np.array([ nx.shortest_path_length(h, frozenset([gene]), root) 
    #    for gene in map(lambda x: int(x), k_shell_numbers.index)])

    # print (np.corrcoef(k_shell_numbers[in_scc]["Kshell number"], 
    #     sps[in_scc]))

    # plt.figure(figsize=[5, 5])
    # plt.scatter(k_shell_numbers[in_scc]["Kshell number"], 
    #     sps[in_scc])
    # plt.xlabel("k shell number")
    # plt.ylabel("merge depth")
    # plt.show()

if __name__ == "__main__":
    main()