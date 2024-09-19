import numpy as np
import matplotlib.pyplot as plt

for sim in range(1, 8):
    alpha_trace = np.load('Lazega/Sim' + str(sim) + 'alphatraceK4.npy')
    plt.plot(alpha_trace.T)
    print(np.mean(alpha_trace, axis=1))
    plt.show()
#pi_trace = np.load('Lazega/Sim1pitraceK3.npy')
