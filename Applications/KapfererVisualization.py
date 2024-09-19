import numpy as np
import matplotlib.pyplot as plt

# 2, 5, 7
#for sim in [2, 5, 7]:
for K in range(2, 20):
    alpha_trace = np.load('Kapferer/Sim' + str(K) + 'alphatraceK' + str(K) + '.npy')
    plt.plot(alpha_trace[:, ::200].T)
    print(np.mean(alpha_trace, axis=1))
    plt.show()
#pi_trace = np.load('Lazega/Sim1pitraceK3.npy')
