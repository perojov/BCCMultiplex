from GibbsSamplers import SymmetricBCMSBMGibbsSamplerSpectral, SymmetricSBMGibbsSampler
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt


D = 3
N = 71
tensor = np.zeros((D, N, N))


with open('Data/Lazega/Lazega-Law-Firm_multiplex.edges') as edgelist:
    edges = edgelist.readlines()
    for edge in edges:
        graph_id, node_from, node_to, _ = edge.split()
        graph_id = int(graph_id) - 1
        node_from = int(node_from) - 1
        node_to = int(node_to) - 1
        tensor[graph_id, node_from, node_to] = tensor[graph_id, node_to, node_from] = 1


sampler = SymmetricSBMGibbsSampler(tensor[0], 20)
sampler.run(1000)
sampler.visualize()

'''

def run_sim(sim_id):
    np.random.seed()
    sampler = SymmetricBCMSBMGibbsSamplerSpectral(tensor, 4)
    sampler.run(50000, 'Lazega/Sim' + str(sim_id))


pool = mp.Pool(mp.cpu_count())
pool.map(run_sim, [sim for sim in range(1, 8)])
'''