import community
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


D = 3
N = 21
multiplex = [nx.Graph() for _ in range(D)]
tensor = np.zeros((D, N, N))

with open('../Data/Krackhardt/Krackhardt-High-Tech_multiplex.edges') as edgelist:
    edges = edgelist.readlines()
    for edge in edges:
        graph_id, node_from, node_to, _ = edge.split()
        graph_id = int(graph_id) - 1
        node_from = int(node_from) - 1
        node_to = int(node_to) - 1
        multiplex[graph_id].add_edge(node_from, node_to)
        tensor[graph_id, node_from, node_to] = tensor[graph_id, node_to, node_from] = 1


def draw_nodes(partition, ax, pos):
    size = float(len(set(partition.values())))
    count = 0.
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=100,
                               node_color=str(count / size), ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)


for d in range(D):
    # first compute the best partition
    G = multiplex[d]

    # Community detection.
    comdet_partition = community.best_partition(G)
    indices_community = list(comdet_partition.values())

    # Stochastic blockmodel.
    sbm_partition = np.load('KrackhardtFinal/Sim0ZK3.npy')[d]
    sbm_dict = {i: sbm_partition[i] for i in range(21)}

    fig, ax = plt.subplots(1, 2)

    pos = nx.spring_layout(G)

    draw_nodes(comdet_partition, ax=ax[0], pos=pos)
    draw_nodes(sbm_dict, ax=ax[1], pos=pos)
    plt.show()
