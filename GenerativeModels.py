import numpy as np
from itertools import product, chain
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
rc('text', usetex=True)
#rc('font', size=17)
#rc('legend', fontsize=17)

#Direct input
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 11,
          }
plt.rcParams.update(params)


class SymmetricBCMultiplexSBM:
    def __init__(self, D=0, K=0, N=0):
        """
        Generative model for the symmetric Bayesian Consensus
        Multiplex Stochastic Blockmodel.
        :param D: Number of layers.
        :param K: Number of classes.
        :param N: Number of nodes.
        """
        self.D, self.K, self.N = D, K, N

        # Hyperparameter configuration.
        self.pi0 = [1] * K
        self.a, self.b = 1, 1
        self.e, self.f = 1, 1

        # Parameters.
        self.pi = np.random.dirichlet(self.pi0)
        self.theta = np.zeros((D, K, K))
        for d in range(self.D):
            for k in range(self.K):
                for l in range(k, self.K):
                    self.theta[d, k, l] = np.random.beta(a=self.a, b=self.b)
                    self.theta[d, l, k] = self.theta[d, k, l]
        self.alpha = np.zeros(self.D)
        for d in range(D):
            val = np.random.beta(self.a, self.b)
            if val < 1 / self.K:
                i = 0
                while val < 1 / self.K:
                    val = np.random.beta(self.a, self.b)
                    i += 1
                    if i > 50:
                        val = 1 / self.K
                        break
            self.alpha[d] = val
        self.C = np.random.choice(a=self.K, size=self.N, p=self.pi)
        self.Z = np.zeros(shape=(self.D, self.N), dtype=np.int)
        for m, i in product(range(self.D), range(self.N)):
            self.Z[m, i] = np.random.choice(self.K, p=[self.v(k, m, i) for k in range(K)])
        print(self.alpha)
        print(self.pi)

    def v(self, k=0, m=0, i=0):
        """
        Adherence function.
        :param k: Cluster to assign.
        :param m: Layer id.
        :param i: Node id.
        :return: Probability.
        """
        if k == self.C[i]:
            return self.alpha[m]
        return (1 - self.alpha[m]) / (self.K - 1)

    def generate(self):
        self.X = np.zeros(shape=(self.D, self.N, self.N))
        for d in range(self.D):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    z_i, z_j = self.Z[d, i], self.Z[d, j]
                    if np.random.uniform() <= self.theta[d, z_i, z_j]:
                        self.X[d, i, j] = self.X[d, j, i] = 1
        return self.X

    def visualize_local(self, X, d, fname):
        """
        Visualizes the stochastic blockmodel on layer m.
        :param m:
        :return:
        """
        blocksize_counter = Counter(self.Z[d])
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.Z[d] == k)[0] for k in block_index]))
        X_blocked = X[:, new_order][new_order]
        sns.heatmap(X_blocked, square=True, cmap='Reds', cbar=False)
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        plt.title(r'Graph %d, $\alpha = %.2f$' % (d + 1, self.alpha[d]))
        for l in np.cumsum([0] + list(blocksize_counter.values())):
            plt.axhline(l, 0, self.N, color='black')
            plt.axvline(l, 0, self.N, color='black')
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()

    def visualize_global(self, X, fname):
        """
        Visualizes the stochastic blockmodel on layer m.
        :param m:
        :return:
        """
        blocksize_counter = Counter(self.C)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.C == k)[0] for k in block_index]))
        X_blocked = X[:, new_order][new_order]
        sns.heatmap(X_blocked, square=True, cmap='Reds', cbar=False)
        frame1 = plt.gca()
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        for l in np.cumsum([0] + list(blocksize_counter.values())):
            plt.axhline(l, 0, self.N, color='black')
            plt.axvline(l, 0, self.N, color='black')
        plt.savefig(fname, bbox_inches='tight')
        plt.clf()

    def visualize(self):
        fix, axs = plt.subplots(2, self.D, sharex='all', sharey='all')
        fix.suptitle('Synthetic multiplex network')

        # Visualization of the local clustering.
        for d in range(self.D):
            blocksize_counter = Counter(self.Z[d])
            block_index = blocksize_counter.keys()
            new_order = list(chain(*[np.where(self.Z[d] == k)[0] for k in block_index]))
            X = self.X[d]
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
            X = self.X[d]
            X_blocked = X[:, new_order][new_order]
            sns.heatmap(X_blocked, square=True, cmap='Blues', cbar=False, ax=axs[1, d])
            for l in np.cumsum([0] + list(blocksize_counter.values())):
                axs[1, d].axhline(l, 0, self.N, color='black')
                axs[1, d].axvline(l, 0, self.N, color='black')
            axs[1, d].set_xticks([])
            axs[1, d].set_yticks([])
            axs[1, d].set_xlabel(r'$\alpha = %.2f$' % self.alpha[d])
        axs[1, 0].set_ylabel("Global \n clustering", rotation=0, ha='right')
        plt.show()


class AsymmetricBernoulliSBM:
    def __init__(self, N, K):
        self.N = N
        self.K = K

        # Edge densities.
        self.theta = np.zeros((K, K))
        for k in range(K):
            for l in range(K):
                self.theta[k, l] = np.random.beta(1, 1)

        # Mixing probability
        self.pi = np.random.dirichlet([1] * K)

        # Graph and latent clustering indices.
        self.X = np.zeros(shape=(self.N, self.N))
        self.Z = np.zeros(shape=self.N)

    def generate(self):
        self.Z = np.random.choice(a=self.K, size=self.N, p=self.pi)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    z_i, z_j = self.Z[i], self.Z[j]
                    if np.random.uniform() <= self.theta[z_i, z_j]:
                        self.X[i, j] = 1
        return self.X

    def visualize(self):
        blocksize_counter = Counter(self.Z)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.Z == k)[0] for k in block_index]))
        X_blocked = self.X[:, new_order][new_order]
        for l in np.cumsum([0] + list(blocksize_counter.values())):
            plt.axhline(l, 0, self.N, color='black')
            plt.axvline(l, 0, self.N, color='black')
        sns.heatmap(X_blocked, square=True, cmap='Blues')
        plt.show()


class SymmetricBernoulliSBM:
    def __init__(self, N, K):
        self.N = N
        self.K = K

        # Edge densities.
        self.theta = np.zeros((K, K))
        for k in range(K):
            for l in range(k, K):
                self.theta[k, l] = self.theta[l, k] = np.random.beta(1/2, 1/2)

        # Mixing probability
        self.pi = np.random.dirichlet([1/2] * K)

        # Graph and latent clustering indices.
        self.X = np.zeros(shape=(self.N, self.N))
        self.Z = np.zeros(shape=self.N)

    def generate(self):
        self.Z = np.random.choice(a=self.K, size=self.N, p=self.pi)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                z_i, z_j = self.Z[i], self.Z[j]
                if np.random.uniform() <= self.theta[z_i, z_j]:
                    self.X[i, j] = self.X[j, i] = 1
        return self.X

    def visualize(self):
        blocksize_counter = Counter(self.Z)
        block_index = blocksize_counter.keys()
        new_order = list(chain(*[np.where(self.Z == k)[0] for k in block_index]))
        X_blocked = self.X[:, new_order][new_order]
        for l in np.cumsum([0] + self.pi):
            plt.axhline(l, 0, self.N, color='black')
            plt.axvline(l, 0, self.N, color='black')
        sns.heatmap(X_blocked, square=True, cmap='Blues')
        plt.show()


class AsymmetricBCMSBM:
    def __init__(self, D=0, K=0, N=0):
        """
        Generative model for the symmetric Bayesian Consensus
        Multiplex Stochastic Blockmodel.
        :param D: Number of layers.
        :param K: Number of classes.
        :param N: Number of nodes.
        """
        self.D, self.K, self.N = D, K, N

        # Parameters.
        self.pi = np.random.dirichlet([1] * self.K)
        self.theta = np.random.beta(1, 1, (self.D, self.K, self.K))
        self.alpha = np.zeros(self.D)
        for d in range(D):
            val = np.random.beta(1, 1)
            if val < 1 / self.K:
                i = 0
                while val < 1 / self.K:
                    val = np.random.beta(1, 1)
                    i += 1
                    if i > 50:
                        val = 1 / self.K
                        break
            self.alpha[d] = val
        self.C = np.random.choice(a=self.K, size=self.N, p=self.pi)
        self.Z = np.zeros(shape=(self.D, self.N), dtype=np.int)
        for m, i in product(range(self.D), range(self.N)):
            self.Z[m, i] = np.random.choice(self.K, p=[self.v(k, m, i) for k in range(K)])

    def v(self, k=0, m=0, i=0):
        """
        Adherence function.
        :param k: Cluster to assign.
        :param m: Layer id.
        :param i: Node id.
        :return: Probability.
        """
        if k == self.C[i]:
            return self.alpha[m]
        return (1 - self.alpha[m]) / (self.K - 1)

    def generate(self):
        self.X = np.zeros(shape=(self.D, self.N, self.N))
        for d in range(self.D):
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        z_i, z_j = self.Z[d, i], self.Z[d, j]
                        if np.random.uniform() <= self.theta[d, z_i, z_j]:
                            self.X[d, i, j] = 1
        return self.X

    def visualize(self):
        fix, axs = plt.subplots(2, self.D, sharex='all', sharey='all')
        fix.suptitle('Synthetic multiplex network')

        # Visualization of the local clustering.
        for d in range(self.D):
            blocksize_counter = Counter(self.Z[d])
            block_index = blocksize_counter.keys()
            new_order = list(chain(*[np.where(self.Z[d] == k)[0] for k in block_index]))
            X = self.X[d]
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
            X = self.X[d]
            X_blocked = X[:, new_order][new_order]
            sns.heatmap(X_blocked, square=True, cmap='Blues', cbar=False, ax=axs[1, d])
            for l in np.cumsum([0] + list(blocksize_counter.values())):
                axs[1, d].axhline(l, 0, self.N, color='black')
                axs[1, d].axvline(l, 0, self.N, color='black')
            axs[1, d].set_xticks([])
            axs[1, d].set_yticks([])
            axs[1, d].set_xlabel(r'$\alpha = %.2f$' % self.alpha[d])
        axs[1, 0].set_ylabel("Global \n clustering", rotation=0, ha='right')
        plt.show()