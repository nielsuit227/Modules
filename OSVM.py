# Version control
# Python 3.6.5
# Numpy 1.16.0
import matplotlib.pyplot as plt
import numpy as np

"""
Online Support Vector Machine.
Sole binary classification, x is nxm with n samples m features. Uses Regularized Follow The Leader from Online Convex
Optimization to optimize the sum of gradient of the squared prediction error with negative entropy regularization in
the setting of the original SVM (Hinge loss + L2 regularization). The data is represented by finite Euclidean balls. 
The representation is updated when the memory is filled with datasamples that don't belong to
the GMM with probability outlierthreshold. Gram matrix not precalculated. 

Couple of options:
shuffle: Shuffles data (recommended for data sets which are sorted for labels)
adaptivekernel: Adapts kernel width based on mean kernel. Standard off, pushes to 0.5 mean.
save_alpha: saves means of multipliers to check convergence, standard off.
save_kernel: saves means of kernels to check convergence of kernel width in case of adaptivekernel

"""


class OSVM(object):
    def __init__(
        self,
        alpha=None,
        xc=None,
        yc=None,
        kernelwidth=0.1,
        learningrate=0.01,
        oca_rad=1,
        regularizer=1,
        shuffle=False,
        adaptivekernel=False,
        save_alpha=False,
    ):
        # Passing parameters
        self._shuffle = shuffle  # Shuffle flag
        self._sig = (
            kernelwidth  # Variance of Gaussian Kernel for Support Vector Machine (SVM)
        )
        self._eta = (
            learningrate  # Learning rate of Regularized Follow The Leader (RFTL)
        )
        self._c = regularizer  # Regularization weight for SVM
        # Online Covering Algorithm
        self._rad = oca_rad  # Radius of representation
        self._oca_l = []  # Online Covering Algorithm - Labels
        self._oca_w = []  # Online Covering Algorithm - Weights
        self._oca_c = []  # Online Covering Algorithm - Centers
        self._k = 1
        # Flags
        self._f_a = save_alpha  # Save mean alpha at each iteration for debug
        self._f_ak = adaptivekernel  # Adapt kernel width for better mean kernel
        # Initialize empty things
        self._alpha = np.array(
            [[0]], dtype="float"
        )  # Decision variable (Lagrangian multipliers)
        self._agrad = np.array([[0]], dtype="float")  # and its derivative
        self._kt = []
        self._amem = np.array([])  # Alpha memory
        # Copy things if
        if alpha is not None:  # Load given alpha
            self._alpha = alpha
            self._agrad = -1 - np.log(alpha)  # Calc belonging sum of gradients (RFTL)
        if xc is not None:
            self._oca_c = xc
            self._oca_l = yc

    # todo, implement multiple kernels
    def _kernel(self, x, y):
        dim_num = x.ndim + y.ndim
        if dim_num == 4:
            if np.min((np.shape(x), np.shape(y))) == 1:
                dim_num = 3
        if dim_num == 2:
            return np.exp(-np.dot(x - y, x - y) / 2 / self._sig**2)
        elif dim_num == 3:
            return np.exp(
                -np.diag(np.dot(x - y, (x - y).T)) / 2 / self._sig**2
            ).reshape((self._k, 1))
        elif dim_num == 4:  # Returns matrix
            gram_matrix = np.zeros((np.size(x, axis=0), np.size(y, axis=0)))
            for i, yi in enumerate(y):
                gram_matrix[i] = self._kernel(x, yi).reshape(self._k)
            return gram_matrix
        else:
            raise Exception(
                "Kernel error: No proper dimensions, tensors not implemented."
            )

    def _oca(self, x, y):
        if len(self._oca_c) == 0:
            self._n = len(x)
            self._oca_c = x.reshape((1, self._n))
            self._oca_l = y.reshape((1, 1))
            self._oca_w = np.array([[1]], dtype="float")
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
        self._alpha = np.vstack(
            (self._alpha, np.zeros((len(self._oca_c) - self._k, 1)))
        )
        self._agrad = np.vstack(
            (self._agrad, np.zeros((len(self._oca_c) - self._k, 1)))
        )
        self._k = len(self._oca_c)
        return

    def update(self, x, y):
        # Representation
        self._oca(x, y)
        # Update
        self._kt = self._kernel(self._oca_c, x)
        if y * np.sum(self._kt * self._alpha * self._oca_l) < 1:
            self._agrad -= y * self._oca_w * self._oca_l * self._kt
            self._alpha = np.minimum(np.exp(-self._eta * self._agrad - 1), self._c)
            if self._oca_l.ndim != 0:
                sa = np.sum(self._alpha)
                self._alpha[self._oca_l == 1] = (
                    self._alpha[self._oca_l == 1]
                    * sa
                    / 2
                    / np.sum(self._alpha[self._oca_l == 1])
                )
                self._alpha[self._oca_l == -1] = (
                    self._alpha[self._oca_l == -1]
                    * sa
                    / 2
                    / np.sum(self._alpha[self._oca_l == -1])
                )

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        max_iter = len(x)
        if self._shuffle:
            ind = np.random.permutation(max_iter)
            x = x[ind]
            y = y[ind]
        for i in range(len(y)):
            print("   [%.0f%%]" % (i * 100 / max_iter), end="\r")
            # Representation
            xt = x[i, :]
            yt = y[i]
            # Update
            self.update(xt, yt)

            # Store mean alpha
            if self._f_a:
                if not self._amem.any():
                    self._amem = np.mean(self._alpha)
                else:
                    self._amem = np.vstack((self._amem, np.mean(self._alpha)))

    # Based on optimized algo predicts new values. Can do multiple at one time.
    def predict(self, xp):
        xp = np.array(xp)
        # Single prediction
        if xp.ndim == 1:
            yp = np.sum(
                self._alpha * self._oca_w * self._oca_l * self._kernel(self._oca_c, xp)
            )
            return yp
        else:  # n-predictions.
            n = np.size(xp, axis=0)
            yp = np.zeros(n)
            for i in range(n):
                yp[i] = np.sum(
                    self._alpha
                    * self._oca_w
                    * self._oca_l
                    * self._kernel(self._oca_c, xp[i, :])
                )
            return yp

    # Visualizes results.
    def visual(
        self,
        grid_size=100,
        levels=10,
        plot_alpha=False,
        plot_kernel=False,
        plot_3=False,
    ):
        cmap = "RdBu_r"
        xmi = 1.5 * np.min(self._oca_c)
        xma = 1.5 * np.max(self._oca_c)
        grid = np.linspace(xmi, xma, grid_size)
        f = np.zeros((grid_size, grid_size))
        for l in range(grid_size):
            for k in range(grid_size):
                f[k, l] = np.sum(
                    self._alpha
                    * self._oca_l
                    * self._oca_w
                    * self._kernel(self._oca_c, np.array([grid[l], grid[k]]))
                )
        plt.figure()
        plt.contour(grid, grid, f, levels=levels, linewidths=0.5, colors="k")
        plt.contourf(grid, grid, f, levels=levels, cmap=cmap)
        plt.suptitle("Online Support Vector Machine")
        plt.colorbar()
        ind = self._oca_l == 1
        plt.scatter(
            self._oca_c[np.where(ind)[0], 0], self._oca_c[np.where(ind)[0], 1], c="r"
        )
        plt.scatter(
            self._oca_c[np.where(~ind)[0], 0], self._oca_c[np.where(~ind)[0], 1], c="b"
        )
        if plot_3 is True:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            x, y = np.meshgrid(grid, grid)
            ax.plot_surface(x, y, f)
        if plot_alpha:
            if not self._amem.any():
                print(
                    "Not plotting Lagrangian convergence, history not saved in optimization."
                )
                print('In order to plot, add "save_alpha=True" to OSVM initialization')
            else:
                plt.figure()
                plt.plot(self._amem)
                plt.title("Convergence of Lagrange multipliers (mean)")
        if plot_kernel:
            if not self._ktmem.any():
                print("Not plotting average kernel, history not saved in optimization.")
                print('In order to plot, add "save_kernel=True" to OSVM initialization')
            else:
                plt.figure()
                plt.plot(self._ktmem)
                plt.suptitle("Average kernel evaluation per iteration")
