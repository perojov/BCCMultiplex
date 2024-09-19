from BCCMultiplex.GenerativeModels import AsymmetricBernoulliSBM
from BCCMultiplex.GibbsSamplers import AsymmetricSBMGibbsSampler
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
