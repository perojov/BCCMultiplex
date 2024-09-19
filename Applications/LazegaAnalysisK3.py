from GibbsSamplers import SymmetricBCMSBMGibbsSampler
import numpy as np
import multiprocessing as mp



D = 3
N = 71
K = 7
'''
tensor = np.zeros((D, N, N))


with open('Data/Lazega/Lazega-Law-Firm_multiplex.edges') as edgelist:
    edges = edgelist.readlines()
    for edge in edges:
        graph_id, node_from, node_to, _ = edge.split()
        graph_id = int(graph_id) - 1
        node_from = int(node_from) - 1
        node_to = int(node_to) - 1
        tensor[graph_id, node_from, node_to] = tensor[graph_id, node_to, node_from] = 1


def run_sim(sim_id):
    np.random.seed()
    sampler = SymmetricBCMSBMGibbsSampler(tensor[:2, :, :], K)
    sampler.run(10000, 'Lazega/Sim10000' + str(sim_id))


pool = mp.Pool(mp.cpu_count())
pool.map(run_sim, [sim for sim in range(1, 4)])
'''


import matplotlib.pyplot as plt

for sim in range(1, 4):
    alpha_trace = np.load('Lazega/Sim10000' + str(sim) + 'alphatraceK' + str(K) + '.npy')
    plt.plot(alpha_trace.T)
    print(np.mean(alpha_trace, axis=1))
    plt.show()

