# Version control
# Python 3.6.5
# Numpy 1.16.0
import numpy as np
import time
import matplotlib.pyplot as plt

'''
Implementation of Online Gradient Descend (OGD) (as there is no additional regularization) to iteratively solve the
Distributionally Robust Optimization (DRO) problem using the kernelized hinge loss.
Uses an Online Covering Algorithm in order to reduce data size. A new point falls under an existing cluster if the
Euclidean distance is smaller than a defined radius (for the same label).

'''


class OCDRO(object):

    def __init__(self,
                 oca_rad=0,
                 regularization=1,
                 wasserstein_rad=0,
                 wasserstein_norm=2,
                 learningrate=0.1,
                 kernelwidth=1,
                 save_alpha=False,
                 save_kt=False,
                 shuffle=False,
                 load_model=None
                 ):
        # Pass parameters
        self._c = regularization
        self._rad = oca_rad
        self._eps = wasserstein_rad
        self._norm = wasserstein_norm
        self._eta = learningrate
        self._sig = kernelwidth
        self._sa = save_alpha
        self._skt = save_kt
        self._shuffle = shuffle
        if save_alpha:
            self._amem = np.array([])
        if save_kt:
            self._ktmem = np.array([])
        # Initialize so PyCharm doesn't whine
        self._oca_l = []  # Online Covering Algorithm - Labels
        self._oca_w = []  # Online Covering Algorithm - Weights
        self._oca_c = []  # Online Covering Algorithm - Centers
        self._k = 1  # Tracker of data set size
        self._n = []  # Feature dimension
        self._alpha = np.array([[0]], dtype='float')  # Decision variable (Lagrangian multipliers)
        self._agrad = np.array([[0]], dtype='float')  # and its derivative
        self._y = []  # Ambiguity set
        self._kt = [] # Kernel vector
        if load_model is not None:
            self._alpha = load_model['multipliers']
            self._oca_w = load_model['oca_weights']
            self._oca_l = load_model['oca_labels']
            self._oca_c = load_model['oca_centers']
            self._sig = load_model['kernel_width']
            self._y = load_model['ambiguity']
            self._k = len(self._alpha)

    def _kernel(self, x, y):
        if x.ndim == 1:
            return np.exp(-np.dot(x - y, x - y) / 2 / self._sig ** 2)
        else:
            return np.exp(-np.diag(np.dot((x - y), (x - y).T)) / 2 / self._sig ** 2).reshape((self._k, 1))

    def _gram(self, x):
        n, m = np.shape(x)
        g = np.zeros((n, n))
        for q in range(n):
            g[q] = self._kernel(x, x[q]).reshape(self._k)
        return g

    def predict(self, x):
        if x.ndim == 1:
            return np.sum(self._alpha * self._oca_w * self._oca_l * self._kernel(self._oca_c - self._y / self._oca_w, x))
        else:
            n, m = np.shape(x)
            yp = np.zeros(n)
            for i, xt in enumerate(x):
                print('       [ %.0f %%]' % (i * 100 / len(x)), end='\r')
                yp[i] = np.sum(self._alpha * self._oca_w * self._oca_l * self._kernel(self._oca_c - self._y / self._oca_w, xt))
            return yp

    def _oca(self, x, y):
        if len(self._oca_c) == 0:
            self._n = len(x)
            self._oca_c = x.reshape((1, self._n))
            self._oca_l = y.reshape((1, 1))
            self._oca_w = np.array([[1]], dtype='float')
            self._y = np.zeros_like(self._oca_c)
            return
        elif y == 1:
            ind = np.where(self._oca_l == 1)[0]
            if len(self._oca_c[ind]) != 0:
                dist = np.sqrt(np.sum((self._oca_c[ind] - x) ** 2, axis=1))
                if (dist < self._rad).any():
                    self._oca_w[ind[dist < self._rad]] += 1 / np.sum(dist < self._rad)
                    return
        elif y == -1:
            ind = np.where(self._oca_l == -1)[0]
            if len(self._oca_c[ind]) != 0:
                dist = np.sqrt(np.sum((self._oca_c[ind] - x) ** 2, axis=1))
                if (dist < self._rad).any():
                    self._oca_w[ind[dist < self._rad]] += 1 / np.sum(dist < self._rad)
                    return
        self._oca_c = np.vstack((self._oca_c, x))
        self._oca_l = np.vstack((self._oca_l, y))
        self._oca_w = np.vstack((self._oca_w, 1))
        self._y = np.vstack((self._y, np.zeros((len(self._oca_c) - self._k, self._n))))
        self._alpha = np.vstack((self._alpha, np.zeros((len(self._oca_c) - self._k, 1))))
        self._agrad = np.vstack((self._agrad, np.zeros((len(self._oca_c) - self._k, 1))))
        self._k = len(self._oca_c)
        return

    def update(self, x, y):
        # Online covering algorithm
        self._oca(x, y)

        # Ambiguity update
        if self._eps != 0:
            self._kt = self._kernel(self._oca_c - self._y / self._oca_w, x)
            self._y -= self._eta * 1 / self._sig ** 2 * y * self._alpha * self._oca_l * self._kt * (
                    self._oca_c - self._y / self._oca_w - x)
            if (np.linalg.norm(self._y, self._norm, axis=1) > self._eps).any():
                ind = np.linalg.norm(self._y, self._norm, axis=1) > self._eps
                self._y[ind] /= (np.linalg.norm(self._y[ind], self._norm, axis=1) / self._eps).reshape((np.sum(ind), 1))

        # Decision update
        self._kt = self._kernel(self._oca_c - self._y / self._oca_w, x)
        if y * np.sum(self._oca_w * self._kt * self._alpha * self._oca_l) < 1:
            self._agrad -= y * self._oca_w * self._oca_l * self._kt
            self._alpha = np.minimum(np.exp(-self._eta * self._agrad - 1), self._c)
            if self._oca_l.ndim != 0:
                sa = np.sum(self._alpha)
                self._alpha[self._oca_l == 1] = self._alpha[self._oca_l == 1] * sa / 2 / np.sum(
                    self._alpha[self._oca_l == 1])
                self._alpha[self._oca_l == -1] = self._alpha[self._oca_l == -1] * sa / 2 / np.sum(
                    self._alpha[self._oca_l == -1])

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        if self._shuffle:
            ind = np.random.permutation(len(x))
            x = x[ind]
            y = y[ind]
        for i in range(len(y)):
            print('       [ %.0f %%]' % (i * 100 / len(y)), end='\r')
            # self._eta = (1 - 1/len(y))*self._eta
            self.update(x[i], y[i])
            if self._sa:
                if not self._amem.any():
                    self._amem = np.mean(self._alpha)
                else:
                    self._amem = np.vstack((self._amem, np.mean(self._alpha)))

    def visual(self, grid_size=100, levels=10, plot_alpha=False, plot_kernel=False, plot_3=False):
        cmap = 'RdBu_r'
        xmi = 1.5 * np.min(self._oca_c)
        xma = 1.5 * np.max(self._oca_c)
        grid = np.linspace(xmi, xma, grid_size)
        f = np.zeros((grid_size, grid_size))
        for l in range(grid_size):
            for k in range(grid_size):
                f[k, l] = np.sum(self._alpha * self._oca_l * self._oca_w * self._kernel(self._oca_c, np.array([grid[l], grid[k]])))
        plt.figure()
        plt.contour(grid, grid, f, levels=levels, linewidths=0.5, colors='k')
        plt.contourf(grid, grid, f, levels=levels, cmap=cmap)
        plt.suptitle('Online Support Vector Machine')
        plt.colorbar()
        ind = self._oca_l == 1
        plt.scatter(self._oca_c[np.where(ind)[0], 0], self._oca_c[np.where(ind)[0], 1], c='r')
        plt.scatter(self._oca_c[np.where(~ind)[0], 0], self._oca_c[np.where(~ind)[0], 1], c='b')
        for i in range(self._k):
            plt.arrow(self._oca_c[i, 0], self._oca_c[i, 1], -self._y[i, 0], -self._y[i, 1], width=0.01)