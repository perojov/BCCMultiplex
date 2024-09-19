import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 15}

rc('font', **font)


def visualize_chains():
    alpha_traces = {}
    alpha_stars = {}
    pi_traces = {}
    for sim in range(7):
        alpha_stars[sim] = np.load('SyntheticD2K3Results/NewestSimulation' + str(sim) + 'alphastarK3.npy')
        alpha_traces[sim] = np.load('SyntheticD2K3Results/NewestSimulation' + str(sim) + 'alphatraceK3.npy')
        pi_traces[sim] = np.load('SyntheticD2K3Results/NewestSimulation' + str(sim) + 'pitraceK3.npy')
        print('Alpha', sim, np.mean(alpha_traces[sim], axis=1))
        print('Pi', sim, np.mean(pi_traces[sim], axis=1))
        plt.plot(pi_traces[sim].T)
        plt.show()


def load_chains(chain_id):
    return np.load('SyntheticD2K3Results/NewestSimulation' + str(chain_id) + 'alphastarK3.npy'), \
           np.load('SyntheticD2K3Results/NewestSimulation' + str(chain_id) + 'alphatraceK3.npy'), \
           np.load('SyntheticD2K3Results/NewestSimulation' + str(chain_id) + 'pitraceK3.npy')


def posterior_alpha_trace_viz():
    '''
    alpha_stars = {}
    alpha_traces = {}
    pi_traces = {}
    for chain_id in [1, 2, 4]:
        alpha_stars[chain_id], alpha_traces[chain_id], pi_traces[chain_id] = load_chains(chain_id)
    '''

    # Rhat values computation
    alpha_traces = {}
    a1_sim = np.zeros((3, 30000))
    a2_sim = np.zeros((3, 30000))
    for id, chain_id in enumerate([1, 2, 4]):
        _, a, _ =  load_chains(chain_id)
        a1_sim[id] = a[0]
        a2_sim[id] = a[1]

    alpha_traces['alpha1'] = a1_sim
    alpha_traces['alpha2'] = a2_sim

    print(az.summary(alpha_traces))

    '''
    # First graph.
    colors = ['red', 'green', 'yellow']
    for id, chain_id in enumerate([1, 2, 4]):
        plt.plot(alpha_traces[chain_id][0, ::40], label='Chain' + str(id + 1), color=colors[id], alpha=0.6)
    length = len(alpha_traces[chain_id][0, ::40])
    plt.plot([0.9] * length, label='True', linewidth=1.5, color='black')
    plt.title(r'Trace plot of $\alpha_1$')
    plt.xlabel('MCMC Interations')
    plt.ylabel(r'$\alpha_1$')
    plt.legend()
    plt.savefig('alpha1trace.pdf', bbox_inches='tight')
    plt.clf()
    '''

    '''
    # Second graph
    colors = ['red', 'green', 'yellow']
    for id, chain_id in enumerate([1, 2, 4]):
        plt.plot(alpha_traces[chain_id][1, ::40], label='Chain' + str(id + 1), color=colors[id], alpha=0.6)
    length = len(alpha_traces[chain_id][1, ::40])
    plt.plot([0.84] * length, label='True', linewidth=1.5, color='black')
    plt.title(r'Trace plot of $\alpha_2$')
    plt.xlabel('MCMC Interations')
    plt.ylabel(r'$\alpha_2$')
    plt.legend()
    plt.savefig('alpha2trace.pdf', bbox_inches='tight')
    '''


def posterior_alpha_density_viz():
    alpha_stars = {}
    alpha_traces = {}
    pi_traces = {}
    alpha1_trace_combined = np.zeros(45000)
    alpha2_trace_combined = np.zeros(45000)
    for i, chain_id in enumerate([1, 2, 4]):
        alpha_stars[chain_id], alpha_traces[chain_id], pi_traces[chain_id] = load_chains(chain_id)
        alpha1_trace_combined[i * 15000: (i + 1) * 15000] = alpha_traces[chain_id][0, 15000:]
        alpha2_trace_combined[i * 15000: (i + 1) * 15000] = alpha_traces[chain_id][1, 15000:]

        # First graph.
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    sns.distplot(alpha1_trace_combined, label=r'$\alpha_1$')
    sns.distplot(alpha2_trace_combined, label=r'$\alpha_2$')
    a1mean = np.mean(alpha1_trace_combined)
    a2mean = np.mean(alpha2_trace_combined)
    true = [0.90159565, 0.84934396]
    a1hpd = az.hpd(alpha1_trace_combined, credible_interval=0.95)
    a2hpd = az.hpd(alpha2_trace_combined, credible_interval=0.95)
    print(a1hpd, a2hpd)
    plt.text(a1mean - 0.05, 8, r'$\bar{\alpha}_1 = %.2f$' % a1mean)
    plt.text(a2mean, 13.5, r'$\bar{\alpha}_2 = %.2f$' % a2mean)
    plt.xlabel('Value')
    plt.title(r'Posterior density of $\alpha$')
    plt.legend()
    plt.savefig('alphadensity.pdf', bbox_inches='tight')


def posterior_pi_trace_viz():
    '''
    alpha_stars = {}
    alpha_traces = {}
    pi_traces = {}
    for chain_id in [1, 2, 4]:
        alpha_stars[chain_id], alpha_traces[chain_id], pi_traces[chain_id] = load_chains(chain_id)
    '''
    # Rhat values computation
    pi_traces = {}
    for chain_id in [1, 2, 4]:
        _, _, pi_traces[chain_id] = load_chains(chain_id)

    rpi_traces = {}

    pi1_sim = np.zeros((3, 30000))
    pi1_sim[0] = pi_traces[1][0]
    pi1_sim[1] = pi_traces[2][0]
    pi1_sim[2] = pi_traces[4][2]

    pi2_sim = np.zeros((3, 30000))
    pi2_sim[0] = pi_traces[1][1]
    pi2_sim[1] = pi_traces[2][2]
    pi2_sim[2] = pi_traces[4][0]

    pi3_sim = np.zeros((3, 30000))
    pi3_sim[0] = pi_traces[1][2]
    pi3_sim[1] = pi_traces[2][1]
    pi3_sim[2] = pi_traces[4][1]

    rpi_traces['pi1'] = pi1_sim
    rpi_traces['pi2'] = pi2_sim
    rpi_traces['pi3'] = pi3_sim

    pi1_rvals = []
    pi2_rvals = []
    pi3_rvals = []
    for i in range(50, 15000, 50):
        first = 0
        last = i
        new_traces = {}
        new_traces['pi1'] = pi1_sim[:, first:last]
        new_traces['pi2'] = pi2_sim[:, first:last]
        new_traces['pi3'] = pi3_sim[:, first:last]
        summary = az.summary(new_traces)
        pi1_rvals.append(summary.loc['pi1', 'r_hat'])
        pi2_rvals.append(summary.loc['pi2', 'r_hat'])
        pi3_rvals.append(summary.loc['pi3', 'r_hat'])
    plt.plot(pi1_rvals, marker='o', markersize=2)
    plt.plot(pi2_rvals, marker='o', markersize=2)
    plt.plot(pi3_rvals, marker='o', markersize=2)
    plt.show()
    #print(az.summary(rpi_traces).loc['pi1', 'r_hat'])
    #print(az.summary(rpi_traces))

    '''
    plt.plot(pi_traces[1][0, ::40], label='Chain' + str(1), color='red', alpha=0.6)
    plt.plot(pi_traces[2][0, ::40], label='Chain' + str(2), color='green', alpha=0.6)
    plt.plot(pi_traces[4][2, ::40], label='Chain' + str(3), color='yellow', alpha=0.6)
    length = len(pi_traces[chain_id][0, ::40])
    plt.plot([0.435] * length, label='True', linewidth=1.5, color='black')
    plt.title(r'Trace plot of $\pi_1$')
    plt.xlabel('MCMC Interations')
    plt.ylabel(r'$\pi_1$')
    plt.legend()
    plt.savefig('pi1trace.pdf', bbox_inches='tight')
    '''
    '''
    plt.plot(pi_traces[1][1, ::40], label='Chain' + str(1), color='red', alpha=0.6)
    plt.plot(pi_traces[2][2, ::40], label='Chain' + str(2), color='green', alpha=0.6)
    plt.plot(pi_traces[4][0, ::40], label='Chain' + str(3), color='yellow', alpha=0.6)
    length = len(pi_traces[chain_id][0, ::40])
    plt.plot([0.03] * length, label='True', linewidth=1.5, color='black')
    plt.title(r'Trace plot of $\pi_2$')
    plt.xlabel('MCMC Interations')
    plt.ylabel(r'$\pi_2$')
    plt.legend()
    plt.savefig('pi2trace.pdf', bbox_inches='tight')
    '''
    '''
    plt.plot(pi_traces[1][2, ::40], label='Chain' + str(1), color='red', alpha=0.6)
    plt.plot(pi_traces[2][1, ::40], label='Chain' + str(2), color='green', alpha=0.6)
    plt.plot(pi_traces[4][1, ::40], label='Chain' + str(3), color='yellow', alpha=0.6)
    length = len(pi_traces[chain_id][0, ::40])
    plt.plot([0.53] * length, label='True', linewidth=1.5, color='black')
    plt.title(r'Trace plot of $\pi_3$')
    plt.xlabel('MCMC Interations')
    plt.ylabel(r'$\pi_3$')
    plt.legend()
    plt.savefig('pi3trace.pdf', bbox_inches='tight')
    '''
    #plt.show()


def posterior_pi_density_viz():
    alpha_stars = {}
    alpha_traces = {}
    pi_traces = {}
    for chain_id in [1, 2, 4]:
        alpha_stars[chain_id], alpha_traces[chain_id], pi_traces[chain_id] = load_chains(chain_id)

    pi1_trace_combined = np.zeros(45000)
    pi2_trace_combined = np.zeros(45000)
    pi3_trace_combined = np.zeros(45000)

    pi1_trace_combined[0:15000] = pi_traces[1][0, 15000:]
    pi1_trace_combined[15000:30000] = pi_traces[2][0, 15000:]
    pi1_trace_combined[30000:45000] = pi_traces[4][2, 15000:]

    pi2_trace_combined[0:15000] = pi_traces[1][1, 15000:]
    pi2_trace_combined[15000:30000] = pi_traces[2][2, 15000:]
    pi2_trace_combined[30000:45000] = pi_traces[4][0, 15000:]

    pi3_trace_combined[0:15000] = pi_traces[1][2, 15000:]
    pi3_trace_combined[15000:30000] = pi_traces[2][1, 15000:]
    pi3_trace_combined[30000:45000] = pi_traces[4][1, 15000:]

    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    sns.distplot(pi1_trace_combined, label=r'$\pi_1$')
    sns.distplot(pi2_trace_combined, label=r'$\pi_2$')
    sns.distplot(pi3_trace_combined, label=r'$\pi_3$')
    pi1hpd = az.hpd(pi1_trace_combined, credible_interval=0.95)
    pi2hpd = az.hpd(pi2_trace_combined, credible_interval=0.95)
    pi3hpd = az.hpd(pi3_trace_combined, credible_interval=0.95)
    print(pi1hpd, pi2hpd, pi3hpd)
    pi1mean = np.mean(pi1_trace_combined)
    pi2mean = np.mean(pi2_trace_combined)
    pi3mean = np.mean(pi3_trace_combined)
    print(pi1mean, pi2mean, pi3mean)
    true = [0.43506731, 0.03103298, 0.53389971]
    plt.text(pi1mean, 8.5, r'$\bar{\pi}_1 = %.2f$' % pi1mean)
    plt.text(pi2mean, 20, r'$\bar{\pi}_2 = %.2f$' % pi2mean)
    plt.text(pi3mean, 8.5, r'$\bar{\pi}_3 = %.2f$' % pi3mean)
    plt.title(r'Posterior density of $\pi$')
    plt.xlabel('Values')
    plt.legend()
    plt.savefig('pidensity.pdf', bbox_inches='tight')
    #plt.show()



posterior_alpha_density_viz()
#posterior_pi_trace_viz()
'''    
    #print(np.mean(alpha_traces[sim][:, 2500:], axis=1))
    #print(np.mean(pi_traces[sim][:, 2500:], axis=1))
# SIM = 2
sim = 2
atrace1 = alpha_traces[sim][0, 2500:]
atrace2 = alpha_traces[sim][1, 2500:]
frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)
sns.distplot(atrace1, label=r'$\alpha_1$')
sns.distplot(atrace2, label=r'$\alpha_2$')
a1mean = np.mean(atrace1)
a2mean = np.mean(atrace2)
# TRUE = [0.54653785 0.60535573]
plt.text(a1mean, 9.5, r'$\bar{\pi}_1 = %.2f$' % a1mean)
plt.text(a2mean, 9.5, r'$\bar{\pi}_2 = %.2f$' % a2mean)
plt.title(r'Posterior density of $\alpha$')
plt.legend()
plt.savefig('alphaDensity.pdf', bbox_inches='tight')
# Highest posterior densities.
a1hpd = az.hpd(atrace1, credible_interval=0.95)
a2hpd = az.hpd(atrace2, credible_interval=0.95)
print(a1hpd)
print(a2hpd)
'''
'''
pitrace1 = alpha_traces[sim][1, 2500:]
pitrace2 = alpha_traces[sim][3, 2500:]
pitrace3 = alpha_traces[sim][2, 2500:]
'''
'''
pi1hpd = az.hpd(pitrace[0], credible_interval=0.95)
pi2hpd = az.hpd(pitrace[1], credible_interval=0.95)
pi3hpd = az.hpd(pitrace[2], credible_interval=0.95)
print(pi1hpd)
print(pi2hpd)
print(pi3hpd)
'''
'''
frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)

sns.distplot(pitrace[0], label=r'$\pi_1$')
sns.distplot(pitrace[1], label=r'$\pi_2$')
sns.distplot(pitrace[2], label=r'$\pi_3$')

pi1mean = np.mean(pitrace[0])
pi2mean = np.mean(pitrace[1])
pi3mean = np.mean(pitrace[2])

plt.text(pi1mean, 9.5, r'$\bar{\pi}_1 = %.2f$' % pi1mean)
plt.text(pi2mean, 9.5, r'$\bar{\pi}_2 = %.2f$' % pi2mean)
plt.text(pi3mean, 19, r'$\bar{\pi}_3 = %.2f$' % pi3mean)
plt.title(r'Posterior density of $\pi$')
plt.legend()
plt.savefig('piDensity.pdf', bbox_inches='tight')
'''
'''
pi1hpd = az.hpd(atrace[0], credible_interval=0.95)
pi2hpd = az.hpd(atrace[1], credible_interval=0.95)
pi3hpd = az.hpd(atrace[2], credible_interval=0.95)

frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)

sns.distplot(atrace[0], label=r'$\alpha_1$')
sns.distplot(atrace[1], label=r'$\alpha_2$')
sns.distplot(atrace[2], label=r'$\alpha_3$')

pi1mean = np.mean(atrace[0])
pi2mean = np.mean(atrace[1])
pi3mean = np.mean(atrace[2])

plt.text(pi1mean, 8.4, r'$\bar{\alpha}_1 = %.2f$' % pi1mean)
plt.text(pi2mean - 0.1, 11.4, r'$\bar{\alpha}_2 = %.2f$' % pi2mean)
plt.text(pi3mean, 12.7, r'$\bar{\alpha}_3 = %.2f$' % pi3mean)
plt.title(r'Posterior density of $\alpha$')
plt.ylim([0, 14])
plt.legend()
plt.savefig('alphaDensity.pdf', bbox_inches='tight')
#plt.show()
'''
