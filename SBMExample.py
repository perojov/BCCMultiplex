from GenerativeModels import AsymmetricBernoulliSBM
from GibbsSamplers import AsymmetricSBMGibbsSampler
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

N = 100
K = 3

model = AsymmetricBernoulliSBM(N, K)
G = model.generate()
model.visualize()
print(model.pi)
sampler = AsymmetricSBMGibbsSampler(G, K)
sampler.run(1000)


'''
def run_model(K):
    np.random.seed()
    #sampler = SymmetricSBMGibbsSampler(X, K)
    #return sampler.run(1000)


pool = mp.Pool(mp.cpu_count())
results = pool.map(run_model, [K for K in range(2, 15)])
print(results)
pool.close()

plt.plot(range(2, 15), results)
plt.show()
'''