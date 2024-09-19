from GibbsSamplers import AsymmetricBCMSBMSampler
from GenerativeModels import AsymmetricBCMSBM
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt


D = 2
K = 3
N = 100

model = AsymmetricBCMSBM(D, K, N)
tensor = model.generate()
print(model.alpha)
print(model.pi)
model.visualize()

sampler = AsymmetricBCMSBMSampler(tensor, K)
sampler.run(20000)


# [0.48063265 0.30818729]
# [0.1831428  0.51454719 0.2139319  0.08837811]