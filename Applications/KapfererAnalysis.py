from GibbsSamplers import SymmetricBCMSBMGibbsSampler
import numpy as np
import multiprocessing as mp


D = 4
N = 39
tensor = np.zeros((D, N, N))


with open('Data/Kapferer/Kapferer-Tailor-Shop_multiplex.edges') as edgelist:
    edges = edgelist.readlines()
    for edge in edges:
        graph_id, node_from, node_to, _ = edge.split()
        graph_id = int(graph_id) - 1
        node_from = int(node_from) - 1
        node_to = int(node_to) - 1
        tensor[graph_id, node_from, node_to] = tensor[graph_id, node_to, node_from] = 1


def run_sim(sim_id):
    np.random.seed()
    sampler = SymmetricBCMSBMGibbsSampler(tensor, sim_id)
    sampler.run(20000, 'Kapferer/Sim' + str(sim_id))


pool = mp.Pool(mp.cpu_count())
pool.map(run_sim, [sim for sim in range(2, 20)])