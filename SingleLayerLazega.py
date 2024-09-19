# Single layer SBM.
from GibbsSamplers import AsymmetricSBMGibbsSampler

# Misc.
import numpy as np
import networkx as nx


def read_network():
    # Adjacency tensor.
    tensor = np.zeros((D, N, N))

    # NetworkX object.
    G = [nx.DiGraph() for _ in range(D)]
    for d in range(D):
        G[d].add_nodes_from(range(21))

    with open('Data/Lazega/Lazega-Law-Firm_multiplex.edges') as edgelist:
        edges = edgelist.readlines()
        for edge in edges:
            graph_id, node_from, node_to, _ = edge.split()
            graph_id = int(graph_id) - 1
            node_from = int(node_from) - 1
            node_to = int(node_to) - 1
            tensor[graph_id, node_from, node_to] = 1
            G[graph_id].add_edge(node_from, node_to)
    return tensor, G


D = 3
N = 71
K = 3

tensor, G = read_network()
sampler = AsymmetricSBMGibbsSampler(tensor[1], K)
sampler.run(10000)