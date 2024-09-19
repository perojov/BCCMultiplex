import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


for sim in range(3):
    a_trace = np.load('KrackhardtFinal/Sim' + str(sim) + 'alphatraceK5.npy')
    plt.plot(a_trace[:, ::40].T)
    plt.show()

'''
astars = []
for K in range(2, 20):
    alpha_star = np.load('KrackhardtModelSelection/SimalphastarK' + str(K) + '.npy')
    astars.append(alpha_star)
    #plt.plot(alpha_trace[:, ::200].T)
    #print(np.mean(alpha_trace, axis=1))
    #plt.show()
#pi_trace = np.load('Lazega/Sim1pitraceK3.npy')
plt.plot(range(2, 20), astars, marker='o', color='blue')

astars = []
for K in range(2, 20):
    alpha_star = np.load('KrackhardtModelSelection/Sim1alphastarK' + str(K) + '.npy')
    astars.append(alpha_star)
    #plt.plot(alpha_trace[:, ::200].T)
    #print(np.mean(alpha_trace, axis=1))
    #plt.show()
#pi_trace = np.load('Lazega/Sim1pitraceK3.npy')
plt.plot(range(2, 20), astars, marker='o', color='red')
'''
'''
astars = []
for K in range(2, 20):
    alpha_trace = np.load('KrackhardtModelSelection/Sim2alphatraceK' + str(K) + '.npy')
    amean = np.mean(alpha_trace[:, 6000:], axis=1)
    astars.append(np.mean((K * amean - 1) / (K - 1)))
    #print(amean)
    #astars.append(alpha_star)
    #plt.plot(alpha_trace[:, ::200].T)
    #print(np.mean(alpha_trace, axis=1))
    #plt.show()
#pi_trace = np.load('Lazega/Sim1pitraceK3.npy')
plt.plot(range(2, 20), astars, marker='o', color='blue')

astars = []
for K in range(2, 20):
    alpha_trace = np.load('KrackhardtModelSelection/Sim3alphatraceK' + str(K) + '.npy')
    amean = np.mean(alpha_trace[:, 6000:], axis=1)
    astars.append(np.mean((K * amean - 1) / (K - 1)))
    #print(amean)
    #astars.append(alpha_star)
    #plt.plot(alpha_trace[:, ::200].T)
    #print(np.mean(alpha_trace, axis=1))
    #plt.show()
#pi_trace = np.load('Lazega/Sim1pitraceK3.npy')
plt.plot(range(2, 20), astars, marker='o', color='purple')

plt.show()
'''