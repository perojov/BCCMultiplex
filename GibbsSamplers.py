import numpy as np
from math import lgamma

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import animation

from collections import Counter
from itertools import chain

rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          }
plt.rcParams.update(params)

log_beta = lambda x, y: lgamma(x) + lgamma(y) - lgamma(x + y)


class AsymmetricBCMSBMSampler:
    def __init__(self, G=np.array([]), K=0):
        """
        The Gibbs sampler for SBM.
        :param G: D x N x N numpy object.
        :param K: Number of classes.
        """
        self.G, self.K = G, K
        self.D, self.N, _ = self.G.shape

        # Neighbors for faster access.
        self.in_neighbors = {}
        self.out_neighbors = {}
        for d in range(self.D):
            for i in range(self.N):
                self.in_neighbors[(d, i)] = np.where(self.G[d, :, i] == 1)[0]
                self.out_neighbors[(d, i)] = np.where(self.G[d, i, :] == 1)[0]

        # Mixing vector.
        self.pi = np.random.dirichlet([1] * self.K)

        # Adherence values.
        self.alpha = np.zeros(self.D)
        for d in range(self.D):
            self.alpha[d] = self.TBeta(1, 1)

        # Global clustering.
        self.C = np.random.choice(a=self.K, size=self.N, p=self.pi)

        # Local clustering.
        self.Z = np.zeros((self.D, self.N), dtype=int)
        for d in range(self.D):
            for i in range(self.N):
                self.Z[d, i] = np.random.choice(self.K,
                                                p=[self.v(k, d, i) for k in range(self.K)])

        # Connectivity strengths.
        self.theta = np.random.beta(1, 1, (self.D, self.K, self.K))

        # Statistics.
        self.T = np.zeros(shape=(self.D, self.K, self.K), dtype=np.int)
        self.R_in = np.zeros(shape=(self.D, self.N, self.K), dtype=np.int)
        self.R_out = np.zeros(shape=(self.D, self.N, self.K), dtype=np.int)
        for d in range(self.D):
            for i in range(self.N):
                for j in range(self.N):
                    self.T[d, self.Z[d, i], self.Z[d, j]] += self.G[d, i, j]
                    self.R_out[d, i, self.Z[d, j]] += self.G[d, i, j]
                    self.R_in[d, i, self.Z[d, j]] += self.G[d, j, i]

        # Local cluster sizes.
        self.n_local = np.zeros(shape=(self.D, self.K), dtype=np.int)
        for d in range(self.D):
            cluster_sizes = Counter(self.Z[d])
            self.n_local[d, list(cluster_sizes.keys())] = list(cluster_sizes.values())

        # Global cluster sizes.
        self.n_global = np.zeros(shape=self.K, dtype=np.int)
        cluster_sizes = Counter(self.C)
        self.n_global[list(cluster_sizes.keys())] = list(cluster_sizes.values())

    def run(self, max_itr=1000, filename=''):
        # Save the traces for post-processing and visualization.
        pi_trace = np.zeros((self.K, max_itr))
        alpha_trace = np.zeros((self.D, max_itr))
        theta_trace = np.zeros((self.D, self.K, self.K, max_itr))

        pi_trace[:, 0] = self.pi
        alpha_trace[:, 0] = self.alpha
        for d in range(self.D):
            theta_trace[d, :, :, 0] = self.theta[d]
        for itr in range(1, max_itr):
            print('Iteration', itr)
            if itr % 2000 == 0:
                self.visualize()
                plt.plot(alpha_trace[:, :itr].T)
                plt.show()
            for d in range(self.D):
                self.theta_posterior(d)
                self.alpha_posterior(d)
                self.Z_posterior(d)
            for i in range(self.N):
                self.C_posterior(i=i)
            self.pi_posterior()

            pi_trace[:, itr] = self.pi
            alpha_trace[:, itr] = self.alpha
            for d in range(self.D):
                theta_trace[d, :, :, itr] = self.theta[d]

        alpha_star = np.zeros(shape=self.D)
        for d in range(self.D):
            alpha = np.mean(alpha_trace[d, int(max_itr / 2):])
            alpha_star[d] = (self.K * alpha - 1) / (self.K - 1)

        #np.save(filename + 'alphastar' + 'K' + str(self._K), np.mean(alpha_star))
        #np.save(filename + 'alphatrace' + 'K' + str(self._K), alpha_trace)
        #np.save(filename + 'pitrace' + 'K' + str(self._K), pi_trace)
        #np.save(filename + 'Z' + 'K' + str(self._K), self.Z)
        #np.save(filename + 'C' + 'K' + str(self._K), self.C)

    def TBeta(self, a=0., b=0.):
        """
        Sample from the Trucanted Beta distribution.
        Truncation is set by K. Sampling from the truncated
        distribution is done by rejection sampling.
        :return:
        """
        val = np.random.beta(a, b)
        if val < 1 / self.K:
            i = 0
            while val < 1 / self.K:
                val = np.random.beta(a, b)
                i += 1
                if i > 50:
                    return 1 / self.K
        return val

    def v(self, k=0, d=0, i=0):
        """
        Adherence function.
        :param k: Cluster to assign.
        :param d: Layer id.
        :param i: Node id.
        :return: Probability.
        """
        if k == self.C[i]:
            return self.alpha[d]
        return (1 - self.alpha[d]) / (self.K - 1)

    def Z_posterior(self, d):
        for i in range(self.N):
            # Remove node i from it's class and take care of statistics.
            self.n_local[d, self.Z[d, i]] -= 1
            self.R_in[d, self.out_neighbors[(d, i)], self.Z[d, i]] -= 1
            self.R_out[d, self.in_neighbors[(d, i)], self.Z[d, i]] -= 1

            self.T[d, self.Z[d, i]] -= self.R_out[d, i]
            self.T[d, :, self.Z[d, i]] -= self.R_in[d, i]

            # Create the probability simplex.
            P = np.log(self.pi)
            aux = np.zeros(self.K)
            for k in range(self.K):
                aux.fill(0)
                aux[k] = 1
                self.n_local[d, k] += 1
                P[k] += np.sum(
                    self.R_out[d, i, :] * np.log(self.theta[d, k, :]) +
                    (self.n_local[d, :] - aux - self.R_out[d, i, :]) * np.log(1 - self.theta[d, k, :]) +
                    self.R_in[d, i, :] * np.log(self.theta[d, :, k]) +
                    (self.n_local[d, :] - aux - self.R_in[d, i, :]) * np.log(1 - self.theta[d, :, k])
                )
                self.n_local[d, k] -= 1

            # Normalize P.
            P = np.exp(P - np.max(P))
            P /= np.sum(P)

            # Sample membership and take care of statistics.
            self.Z[d, i] = np.random.choice(self.K, p=P)
            self.n_local[d, self.Z[d, i]] += 1
            self.R_in[d, self.out_neighbors[(d, i)], self.Z[d, i]] += 1
            self.R_out[d, self.in_neighbors[(d, i)], self.Z[d, i]] += 1

            self.T[d, self.Z[d, i]] += self.R_out[d, i]
            self.T[d, :, self.Z[d, i]] += self.R_in[d, i]

    def theta_posterior(self, d=0):
        for k in range(self.K):
            for l in range(self.K):
                self.theta[d, k, l] = np.random.beta(
                    self.T[d, k, l] + 1,
                    self.n_local[d, k] * (self.n_local[d, l] - (k == l)) - self.T[d, k, l] + 1
                )

    def pi_posterior(self):
        self.pi = np.random.dirichlet(self.n_global + 1)

    def alpha_posterior(self, d=0):
        """
        Sample from the posterior of alpha_d
        :param d: Graph index.
        :return:
        """
        tau = int(np.sum(self.Z[d] == self.C))
        self.alpha[d] = self.TBeta(tau + 1, self.N - tau + 1)

    def C_posterior(self, i=0):
        self.n_global[self.C[i]] -= 1
        P = np.log(self.pi)
        for k in range(self.K):
            self.C[i] = k
            for d in range(self.D):
                P[k] += np.log(self.v(self.Z[d, i], d, i))
        P = np.exp(P - np.max(P))
        P /= np.sum(P)
        self.C[i] = np.random.choice(self.K, p=P)
        self.n_global[self.C[i]] += 1

    def visualize(self):

        fix, axs = plt.subplots(2, self.D, sharex='all', sharey='all')
        fix.suptitle('Learned source-specific and consensus clustering')

        # Visualization of the local clustering.
        for d in range(self.D):
            blocksize_counter = Counter(self.Z[d])
            block_index = blocksize_counter.keys()
            new_order = list(chain(*[np.where(self.Z[d] == k)[0] for k in block_index]))
            X = self.G[d]
            X_blocked = X[:, new_order][new_order]
            sns.heatmap(X_blocked, square=True, cmap='Blues', cbar=False, ax=axs[0, d])
            for l in np.cumsum([0] + list(blocksize_counter.values())):
                axs[0, d].axhline(l, 0, self.N, color='black')
                axs[0, d].axvline(l, 0, self.N, color='black')
            axs[0, d].set_title('Graph' + str(d + 1))
            axs[0, d].set_xticks([])
            axs[0, d].set_yticks([])
        axs[0, 0].set_ylabel("Local \n clustering", rotation=0, ha='right')
        # Visualization of global clustering.
        blocksize_counter = Counter(self.C)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.C == k)[0] for k in block_index]))
        for d in range(self.D):
            X = self.G[d]
            X_blocked = X[:, new_order][new_order]
            sns.heatmap(X_blocked, square=True, cmap='Blues', cbar=False, ax=axs[1, d])
            for l in np.cumsum([0] + list(blocksize_counter.values())):
                axs[1, d].axhline(l, 0, self.N, color='black')
                axs[1, d].axvline(l, 0, self.N, color='black')
            axs[1, d].set_xticks([])
            axs[1, d].set_yticks([])
            #axs[1, d].set_xlabel(r'$\alpha = %.2f$' % np.mean)
        axs[1, 0].set_ylabel("Global \n clustering", rotation=0, ha='right')
        plt.show()


class AsymmetricSBMGibbsSampler:
    def __init__(self, G=np.array([]), K=0):
        """
        The Gibbs sampler for SBM.
        :param G: N x N numpy object.
        :param K: Number of classes.
        """
        self._N, self._K, self._G = G.shape[0], K, G

        # Neighbors.
        self.in_neighbors = {}
        self.out_neighbors = {}
        for i in range(self._N):
            self.in_neighbors[i] = np.where(self._G[:, i] == 1)[0]
            self.out_neighbors[i] = np.where(self._G[i, :] == 1)[0]

        # Mixing vector.
        self.pi = np.random.dirichlet([1] * K)

        # Local clustering.
        self.Z = np.random.choice(self._K, self._N, p=self.pi)

        # Connectivity strength.
        self.theta = np.zeros(shape=(self._K, self._K))
        for k in range(self._K):
            for l in range(self._K):
                self.theta[k, l] = np.random.beta(a=1, b=1)

        # Sufficient statistics.
        self.T = np.zeros(shape=(self._K, self._K), dtype=np.int)
        self.R_in = np.zeros(shape=(self._N, self._K), dtype=np.int)
        self.R_out = np.zeros(shape=(self._N, self._K), dtype=np.int)
        for i in range(self._N):
            for j in range(self._N):
                self.T[self.Z[i], self.Z[j]] += self._G[i, j]
                self.R_out[i, self.Z[j]] += self._G[i, j]
                self.R_in[i, self.Z[j]] += self._G[j, i]

        # Cluster sizes.
        self.n = np.zeros(shape=self._K, dtype=np.int)
        cluster_sizes = Counter(self.Z)
        self.n[list(cluster_sizes.keys())] = list(cluster_sizes.values())

    def Z_posterior(self):
        for i in range(self._N):
            # Remove node i from it's class and take care of statistics.
            self.n[self.Z[i]] -= 1
            self.R_in[self.out_neighbors[i], self.Z[i]] -= 1
            self.R_out[self.in_neighbors[i], self.Z[i]] -= 1

            self.T[self.Z[i]] -= self.R_out[i]
            self.T[:, self.Z[i]] -= self.R_in[i]

            # Create the probability simplex.
            P = np.log(self.pi)
            aux = np.zeros(self._K)
            for k in range(self._K):
                aux.fill(0)
                aux[k] = 1
                self.n[k] += 1
                P[k] += np.sum(
                    self.R_out[i, :] * np.log(self.theta[k, :]) +
                    (self.n - aux - self.R_out[i, :]) * np.log(1 - self.theta[k, :]) +
                    self.R_in[i, :] * np.log(self.theta[:, k]) +
                    (self.n - aux - self.R_in[i, :]) * np.log(1 - self.theta[:, k])
                )
                self.n[k] -= 1

            # Normalize P.
            P = np.exp(P - np.max(P))
            P /= np.sum(P)

            # Sample membership and take care of statistics.
            self.Z[i] = np.random.choice(self._K, p=P)
            self.n[self.Z[i]] += 1
            self.R_in[self.out_neighbors[i], self.Z[i]] += 1
            self.R_out[self.in_neighbors[i], self.Z[i]] += 1

            self.T[self.Z[i]] += self.R_out[i]
            self.T[:, self.Z[i]] += self.R_in[i]

    def theta_posterior(self):
        for k in range(self._K):
            for l in range(self._K):
                self.theta[k, l] = np.random.beta(
                    self.T[k, l] + 1,
                    (self.n[k] * (self.n[l] - (k == l))) - self.T[k, l] + 1
                )

    def pi_posterior(self):
        self.pi = np.random.dirichlet(self.n + 1)

    def run(self, max_itr=1000, filename=''):
        # Save the traces for post-processing and visualization.
        pi_trace = np.zeros((self._K, max_itr))
        theta_trace = np.zeros((self._K, self._K, max_itr))

        pi_trace[:, 0] = self.pi
        theta_trace[:, :, 0] = self.theta
        for itr in range(1, max_itr):
            print('Iteration', itr)
            self.theta_posterior()
            self.Z_posterior()
            self.pi_posterior()
            pi_trace[:, itr] = self.pi
            theta_trace[:, :, itr] = self.theta
        print(np.mean(pi_trace, axis=1))
        plt.plot(pi_trace.T)
        plt.show()

    def visualize(self, it):
        blocksize_counter = Counter(self.Z)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.Z == k)[0] for k in block_index]))
        X_blocked = self._G[:, new_order][new_order]
        for l in np.cumsum([0] + list(blocksize_counter.values())):
            plt.axhline(l, 0, self._N, color='black')
            plt.axvline(l, 0, self._N, color='black')
        sns.heatmap(X_blocked, square=True, cmap='Blues')
        plt.show()


class SymmetricSBMGibbsSampler:
    def __init__(self, g=np.array([]), K=0):
        """
        The Gibbs sampler for SBM.
        :param g: N x N numpy object.
        :param K: Number of classes.
        """
        self._N, self._K, self._g = g.shape[0], K, g
        self.neighbors = {}
        for i in range(self._N):
            self.neighbors[i] = np.where(self._g[i] == 1)[0]

        self.pi = np.random.dirichlet([1] * K)

        self.Z = np.random.choice(self._K, self._N, p=self.pi)

        self.theta = np.zeros(shape=(self._K, self._K))
        for k in range(self._K):
            for l in range(k, self._K):
                self.theta[k, l] = np.random.beta(a=1, b=1)
                self.theta[l, k] = self.theta[k, l]

        self.T = np.zeros(shape=(self._K, self._K), dtype=np.int)
        self.R = np.zeros(shape=(self._N, self._K), dtype=np.int)
        for i in range(self._N):
            for j in range(self._N):
                self.T[self.Z[i], self.Z[j]] += self._g[i, j]
                self.R[i, self.Z[j]] += self._g[i, j]
        np.fill_diagonal(self.T, self.T.diagonal() / 2.)

        self.n = np.zeros(shape=self._K, dtype=np.int)
        cluster_sizes = Counter(self.Z)
        self.n[list(cluster_sizes.keys())] = list(cluster_sizes.values())

    def run(self, max_itr=1000, filename=''):
        # Save the traces for post-processing and visualization.
        pi_trace = np.zeros((self._K, max_itr))
        theta_trace = np.zeros((self._K, self._K, max_itr))

        pi_trace[:, 0] = self.pi
        theta_trace[:, :, 0] = self.theta

        for itr in range(1, max_itr):
            print('Iteration', itr)
            self.theta_posterior()
            self.Z_posterior()
            self.pi_posterior()
            pi_trace[:, itr] = self.pi
            theta_trace[:, :, itr] = self.theta
        plt.plot(pi_trace.T)
        plt.show()
        return self.Z

    def log_likelihood(self):
        log_lik = 0.0
        for k in range(self._K):
            for l in range(k, self._K):
                log_lik += self.T[k, l] * np.log(self.theta[k, l] / (1 - self.theta[k, l]))
                if k != l:
                    log_lik += np.log(1 - self.theta[k, l]) * self.n[k] * self.n[l]
                else:
                    log_lik += np.log(1 - self.theta[k, l]) * self.n[k] * (self.n[k] - 1) / 2
        for k in range(self._K):
            log_lik += self.n[k] * np.log(self.pi[k])
            log_lik += (1/2 - 1) * np.log(self.pi[k])
        for k in range(self._K):
            for l in range(k, self._K):
                log_lik += (1 / 2 - 1) * np.log(self.theta[k, l]) + (1 / 2 - 1) * np.log(1 - self.theta[k, l])
        return log_lik

    def Z_posterior(self):
        for i in range(self._N):
            # Remove node i from it's class and take care of statistics.
            self.n[self.Z[i]] -= 1
            self.R[self.neighbors[i], self.Z[i]] -= 1
            self.T[self.Z[i]] -= self.R[i]
            self.T[:, self.Z[i]] = self.T[self.Z[i]]

            # Create the probability simplex.
            P = np.log(self.pi)
            aux = np.zeros(self._K)
            for k in range(self._K):
                aux.fill(0)
                aux[k] = 1
                self.n[k] += 1
                P[k] += np.sum(
                    self.R[i, :] * np.log(self.theta[k, :]) +
                    (self.n - aux - self.R[i, :]) * np.log(1 - self.theta[k, :])
                )
                self.n[k] -= 1

            # Normalize P.
            P = np.exp(P - np.max(P))
            P /= np.sum(P)

            # Sample membership and take care of statistics.
            self.Z[i] = np.random.choice(self._K, p=P)
            self.n[self.Z[i]] += 1
            self.R[self.neighbors[i], self.Z[i]] += 1
            self.T[self.Z[i]] += self.R[i]
            self.T[:, self.Z[i]] = self.T[self.Z[i]]

    def theta_posterior(self):
        for k in range(self._K):
            self.theta[k, k] = np.random.beta(self.T[k, k] + 1,
                                              (self.n[k] * (self.n[k] - 1)) / 2 - self.T[k, k] + 1)
        for k in range(self._K):
            for l in range(k + 1, self._K):
                self.theta[k, l] = np.random.beta(self.T[k, l] + 1,
                                                  self.n[k] * self.n[l] - self.T[k, l] + 1)
                self.theta[l, k] = self.theta[k, l]

    def pi_posterior(self):
        self.pi = np.random.dirichlet(self.n + 1)

    def visualize(self):
        blocksize_counter = Counter(self.Z)
        print(blocksize_counter)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.Z == k)[0] for k in block_index]))
        X_blocked = self._g[:, new_order][new_order]
        for l in np.cumsum([0] + list(blocksize_counter.values())):
            plt.axhline(l, 0, self._N, color='black')
            plt.axvline(l, 0, self._N, color='black')
        sns.heatmap(X_blocked, square=True, cmap='Blues')
        plt.show()


class SymmetricSBMCollapsedGibbsSampler:
    def __init__(self, g=np.array([]), K=0):
        """
        The Gibbs sampler for SBM.
        :param g: N x N numpy array.
        :param K: Number of classes.
        """
        self._N, self._K, self._g = g.shape[0], K, g
        self.neighbors = {}
        for i in range(self._N):
            self.neighbors[i] = np.where(self._g[i] == 1)[0]
        self.pi = np.random.dirichlet([1] * K)
        self.Z = np.random.choice(self._K, self._N, p=self.pi)
        self.T = np.zeros(shape=(self._K, self._K), dtype=np.int)
        self.R = np.zeros(shape=(self._N, self._K), dtype=np.int)
        for i in range(self._N):
            for j in range(self._N):
                self.T[self.Z[i], self.Z[j]] += self._g[i, j]
                self.R[i, self.Z[j]] += self._g[i, j]
        np.fill_diagonal(self.T, self.T.diagonal() / 2.)
        self.n = np.zeros(shape=self._K, dtype=np.int)
        cluster_sizes = Counter(self.Z)
        self.n[list(cluster_sizes.keys())] = list(cluster_sizes.values())

    def run(self, max_itr=1000, filename=''):
        # Save the traces for post-processing and visualization.
        pi_trace = np.zeros((self._K, max_itr))
        pi_trace[:, 0] = self.pi

        for itr in range(1, max_itr):
            print('Iteration', itr)
            if itr % 300 == 0:
                self.visualize()
            self.Z_posterior()
            self.pi_posterior()
            pi_trace[:, itr] = self.pi

        plt.plot(pi_trace.T)
        print(np.mean(pi_trace, axis=1))
        plt.show()

    def Z_posterior(self):
        for i in range(self._N):
            # Remove node i from it's class and take care of statistics.
            self.n[self.Z[i]] -= 1
            self.R[self.neighbors[i], self.Z[i]] -= 1
            self.T[self.Z[i]] -= self.R[i]
            self.T[:, self.Z[i]] = self.T[self.Z[i]]

            # Create the probability simplex.
            P = np.log(self.pi)
            for k in range(self._K):
                self.n[k] += 1
                for l in range(self._K):
                    if k == l:
                        Nmin = (self.n[k] * (self.n[k] - 1)) / 2 - self.T[k, k]
                    else:
                        Nmin = self.n[k] * self.n[l] - self.T[k, l]
                    P[k] += log_beta(
                        self.T[k, l] + self.R[i, l] + 1 / 2,
                        Nmin + self.n[l] - (k == l) - self.R[i, l] + 1 / 2
                    ) - log_beta(self.T[k, l] + 1 / 2, Nmin + 1 / 2)
                self.n[k] -= 1
            # Normalize P.
            P = np.exp(P - np.max(P))
            P /= np.sum(P)

            # Sample membership and take care of statistics.
            self.Z[i] = np.random.choice(self._K, p=P)
            self.n[self.Z[i]] += 1
            self.R[self.neighbors[i], self.Z[i]] += 1
            self.T[self.Z[i]] += self.R[i]
            self.T[:, self.Z[i]] = self.T[self.Z[i]]

    def pi_posterior(self):
        self.pi = np.random.dirichlet(self.n + 1 / 2)

    def visualize(self):
        blocksize_counter = Counter(self.Z)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.Z == k)[0] for k in block_index]))
        X_blocked = self._g[:, new_order][new_order]
        for l in np.cumsum([0] + self.pi):
            plt.axhline(l, 0, self._N, color='black')
            plt.axvline(l, 0, self._N, color='black')
        #sns.heatmap(X_blocked, square=True, cmap='Blues')
        plt.imshow(X_blocked)
        plt.show()


class SymmetricBCMSBMGibbsSampler:
    def __init__(self, G=np.array([]), K=0):
        """
        The Gibbs sampler for SBM.
        :param G: D x N x N numpy object.
        :param K: Number of classes.
        """
        self._G, self._K = G, K
        self._D, self._N, _ = self._G.shape

        # Neighbors for faster access.
        self.neighbors = {}
        for d in range(self._D):
            for i in range(self._N):
                self.neighbors[(d, i)] = np.where(self._G[d, i, :] == 1)[0]

        # Mixing vector.
        self.pi = np.random.dirichlet([1] * K)

        # Adherence values.
        self.alpha = np.zeros(self._D)
        for d in range(self._D):
            self.alpha[d] = self.TBeta(1, 1)

        # Global clustering.
        self.C = np.random.choice(a=self._K, size=self._N, p=self.pi)

        # Local clustering.
        self.Z = np.zeros((self._D, self._N), dtype=int)
        for d in range(self._D):
            for i in range(self._N):
                self.Z[d, i] = np.random.choice(self._K, p=[self.v(k, d, i) for k in range(self._K)])

        # Connectivity strengths.
        self.theta = np.zeros(shape=(self._D, self._K, self._K))
        for d in range(self._D):
            for k in range(self._K):
                for l in range(k, self._K):
                    self.theta[d, k, l] = np.random.beta(a=1, b=1)
                    self.theta[d, l, k] = self.theta[d, k, l]

        self.T = np.zeros(shape=(self._D, self._K, self._K), dtype=np.int)
        self.R = np.zeros(shape=(self._D, self._N, self._K), dtype=np.int)
        for d in range(self._D):
            for i in range(self._N):
                for j in range(self._N):
                    self.T[d, self.Z[d, i], self.Z[d, j]] += self._G[d, i, j]
                    self.R[d, i, self.Z[d, j]] += self._G[d, i, j]
            np.fill_diagonal(self.T[d, :, :], self.T[d, :, :].diagonal() / 2.)
        self.n_local = np.zeros(shape=(self._D, self._K), dtype=np.int)
        for d in range(self._D):
            cluster_sizes = Counter(self.Z[d])
            self.n_local[d, list(cluster_sizes.keys())] = list(cluster_sizes.values())
        self.n_global = np.zeros(shape=self._K, dtype=np.int)
        cluster_sizes = Counter(self.C)
        self.n_global[list(cluster_sizes.keys())] = list(cluster_sizes.values())

    def TBeta(self, a=0., b=0.):
        """
        Sample from the Trucanted Beta distribution.
        Truncation is set by K. Sampling from the truncated
        distribution is done by rejection sampling.
        :return:
        """
        val = np.random.beta(a, b)
        if val < 1 / self._K:
            i = 0
            while val < 1 / self._K:
                val = np.random.beta(a, b)
                i += 1
                if i > 40:
                    return 1 / self._K
        return val

    def v(self, k=0, d=0, i=0):
        """
        Adherence function.
        :param k: Cluster to assign.
        :param d: Layer id.
        :param i: Node id.
        :return: Probability.
        """
        if k == self.C[i]:
            return self.alpha[d]
        return (1 - self.alpha[d]) / (self._K - 1)

    def run(self, max_itr=1000, filename=''):
        # Save the traces for post-processing and visualization.
        pi_trace = np.zeros((self._K, max_itr))
        alpha_trace = np.zeros((self._D, max_itr))
        theta_trace = np.zeros((self._D, self._K, self._K, max_itr))
        alpha_star_trace = np.zeros((self._D, max_itr))

        pi_trace[:, 0] = self.pi
        alpha_trace[:, 0] = self.alpha
        for d in range(self._D):
            theta_trace[d, :, :, 0] = self.theta[d]
            alpha_star_trace[d, 0] = (self._K * self.alpha[d] - 1) / (self._K - 1)

        for itr in range(1, max_itr):
            print('Iteration', itr)
            if itr % 10000 == 0:
                self.visualize()
                print('pi=', np.mean(pi_trace[:, :itr], axis=1))
                print('alpha=', np.mean(alpha_trace[:, :itr], axis=1))
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(alpha_trace[:, :itr].T)
                ax[0].set_title(r'$\alpha$')
                ax[1].plot(pi_trace[:, :itr].T)
                ax[1].set_title(r'$\pi$')
                plt.show()

            for d in range(self._D):
                self.theta_posterior(d)
                self.alpha_posterior(d)
                self.Z_posterior(d)

            for i in range(self._N):
                self.C_posterior(i=i)
            self.pi_posterior()

            pi_trace[:, itr] = self.pi
            alpha_trace[:, itr] = self.alpha
            for d in range(self._D):
                theta_trace[d, :, :, itr] = self.theta[d]
                alpha_star_trace[d, itr] = (self._K * self.alpha[d] - 1) / (self._K - 1)

        alpha_star = np.zeros(shape=self._D)
        for d in range(self._D):
            alpha = np.mean(alpha_trace[d, int(max_itr / 2):-1])
            alpha_star[d] = (self._K * alpha - 1) / (self._K - 1)

        np.save(filename + 'alphastar' + 'K' + str(self._K), np.mean(alpha_star))
        np.save(filename + 'alphatrace' + 'K' + str(self._K), alpha_trace)
        np.save(filename + 'pitrace' + 'K' + str(self._K), pi_trace)
        np.save(filename + 'alphastartrace' + 'K' + str(self._K), alpha_star_trace)
        np.save(filename + 'Z' + 'K' + str(self._K), self.Z)
        np.save(filename + 'C' + 'K' + str(self._K), self.C)

    def Z_posterior(self, d):
        for i in range(self._N):
            # Remove node i from it's class and take care of statistics.
            self.n_local[d, self.Z[d, i]] -= 1
            self.R[d, self.neighbors[(d, i)], self.Z[d, i]] -= 1
            self.T[d, self.Z[d, i]] -= self.R[d, i]
            self.T[d, :, self.Z[d, i]] = self.T[d, self.Z[d, i]]

            # Create the probability simplex.
            P = np.array([np.log(self.v(k, d, i)) for k in range(self._K)])
            aux = np.zeros(self._K)
            for k in range(self._K):
                aux.fill(0)
                aux[k] = 1
                self.n_local[d, k] += 1
                P[k] += np.sum(
                    self.R[d, i, :] * np.log(self.theta[d, k, :]) +
                    (self.n_local[d, :] - aux - self.R[d, i, :]) * np.log(1 - self.theta[d, k, :])
                )
                self.n_local[d, k] -= 1

            # Normalize P.
            P = np.exp(P - np.max(P))
            P /= np.sum(P)

            # Sample membership and take care of statistics.
            self.Z[d, i] = np.random.choice(self._K, p=P)
            self.n_local[d, self.Z[d, i]] += 1
            self.R[d, self.neighbors[(d, i)], self.Z[d, i]] += 1
            self.T[d, self.Z[d, i]] += self.R[d, i]
            self.T[d, :, self.Z[d, i]] = self.T[d, self.Z[d, i]]

    def theta_posterior(self, d=0):
        for k in range(self._K):
            Nmin = (self.n_local[d, k] * (self.n_local[d, k] - 1)) / 2
            self.theta[d, k, k] = np.random.beta(
                self.T[d, k, k] + 1, Nmin - self.T[d, k, k] + 1
            )
        for k in range(self._K):
            for l in range(k + 1, self._K):
                Nmin = self.n_local[d, k] * self.n_local[d, l]
                self.theta[d, k, l] = np.random.beta(
                    self.T[d, k, l] + 1, Nmin - self.T[d, k, l] + 1
                )
                self.theta[d, l, k] = self.theta[d, k, l]

    def pi_posterior(self):
        self.pi = np.random.dirichlet(self.n_global + 1)

    def alpha_posterior(self, d=0):
        """
        Sample from the posterior of alpha_d
        :param d: Graph index.
        :return:
        """
        tau = int(np.sum(self.Z[d] == self.C))
        self.alpha[d] = self.TBeta(tau + 1, self._N - tau + 1)

    def C_posterior(self, i=0):
        self.n_global[self.C[i]] -= 1
        P = np.log(self.pi)
        for k in range(self._K):
            self.C[i] = k
            for d in range(self._D):
                P[k] += np.log(self.v(self.Z[d, i], d, i))
        P = np.exp(P - np.max(P))
        P /= np.sum(P)
        self.C[i] = np.random.choice(self._K, p=P)
        self.n_global[self.C[i]] += 1

    def visualize(self):

        fix, axs = plt.subplots(2, self._D, sharex='all', sharey='all')
        fix.suptitle('Learned source-specific and consensus clustering')

        # Visualization of the local clustering.
        for d in range(self._D):
            blocksize_counter = Counter(self.Z[d])
            block_index = blocksize_counter.keys()
            new_order = list(chain(*[np.where(self.Z[d] == k)[0] for k in block_index]))
            X = self._G[d]
            X_blocked = X[:, new_order][new_order]
            sns.heatmap(X_blocked, square=True, cmap='Blues', cbar=False, ax=axs[0, d])
            for l in np.cumsum([0] + list(blocksize_counter.values())):
                axs[0, d].axhline(l, 0, self._N, color='black')
                axs[0, d].axvline(l, 0, self._N, color='black')
            axs[0, d].set_title('Graph' + str(d + 1))
            axs[0, d].set_xticks([])
            axs[0, d].set_yticks([])
        axs[0, 0].set_ylabel("Local \n clustering", rotation=0, ha='right')
        # Visualization of global clustering.
        blocksize_counter = Counter(self.C)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.C == k)[0] for k in block_index]))
        for d in range(self._D):
            X = self._G[d]
            X_blocked = X[:, new_order][new_order]
            sns.heatmap(X_blocked, square=True, cmap='Blues', cbar=False, ax=axs[1, d])
            for l in np.cumsum([0] + list(blocksize_counter.values())):
                axs[1, d].axhline(l, 0, self._N, color='black')
                axs[1, d].axvline(l, 0, self._N, color='black')
            axs[1, d].set_xticks([])
            axs[1, d].set_yticks([])
            #axs[1, d].set_xlabel(r'$\alpha = %.2f$' % np.mean)
        axs[1, 0].set_ylabel("Global \n clustering", rotation=0, ha='right')
        plt.show()


