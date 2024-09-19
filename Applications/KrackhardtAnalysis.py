# Single layer SBM.
from BCCMultiplex.GibbsSamplers import AsymmetricSBMGibbsSampler

# Multiplex consensus clustering.
from BCCMultiplex.GibbsSamplers import SymmetricBCMSBMGibbsSampler

# Misc.
import numpy as np
import networkx as nx
import multiprocessing as mp
import community
from collections import Counter
from itertools import chain
import seaborn as sns
import matplotlib.pyplot as plt


def read_network():
    # Adjacency tensor.
    tensor = np.zeros((D, N, N))

    # NetworkX object.
    G = [nx.DiGraph() for _ in range(D)]
    for d in range(D):
        G[d].add_nodes_from(range(21))

    #with open('Data/Krackhardt/Krackhardt-High-Tech_multiplex.edges') as edgelist:
    with open('../Data/Lazega/Lazega-Law-Firm_multiplex.edges') as edgelist:
        edges = edgelist.readlines()
        for edge in edges:
            graph_id, node_from, node_to, _ = edge.split()
            graph_id = int(graph_id) - 1
            node_from = int(node_from) - 1
            node_to = int(node_to) - 1
            tensor[graph_id, node_from, node_to] = 1
            G[graph_id].add_edge(node_from, node_to)
    return tensor, G


def visualize_network(network, Z):
    if isinstance(Z, dict):
        Z = np.array(list(dict(sorted(Z.items())).values()))
    N = network.shape[0]
    blocksize_counter = Counter(Z)
    block_index = blocksize_counter.keys()
    new_order = list(chain(*[np.where(Z == k)[0] for k in block_index]))
    X_blocked = network[:, new_order][new_order]
    for l in np.cumsum([0] + list(blocksize_counter.values())):
        plt.axhline(l, 0, N, color='black')
        plt.axvline(l, 0, N, color='black')
    sns.heatmap(X_blocked, square=True, cmap='Blues')
    plt.show()


def single_layer_analysis():
    for d in range(D):
        partition = community.best_partition(G[d])
        sbm_sampler = AsymmetricSBMGibbsSampler(tensor[d], K=K)
        Z = sbm_sampler.run(10000)
        visualize_network(tensor[d], partition)
        visualize_network(tensor[d], Z)


def consensus_analysis(sim):
    np.random.seed()
    sampler = SymmetricBCMSBMGibbsSampler(tensor, K)
    sampler.run(100000, 'KrackhardtFinal/Sim' + str(sim))


# Network spec.
D = 3
N = 71
K = 4

tensor, G = read_network()
single_layer_analysis()
'''
pool = mp.Pool(mp.cpu_count())
pool.map(consensus_analysis, [sim for sim in range(3)])
'''
