# Libraries
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as skGMM


class OSVC(object):
    def __init__(
        self,
        multipliers=100,
        kernel_width=2.0,
        learning_rate=0.001,
        max_int_pdf=100.0,
        memory_size=5000,
        outlier_threshold=0.2,
        selection_size=5000,
        ac=None,
        xc=None,
    ):
        self._mp = multipliers
        if ac is None:
            self._alpha = np.zeros(self._mp)
        else:
            self._alpha = ac
        if xc is not None:
            self._xc = xc
        self._m = []
        self._memorySize = memory_size
        self._outlierThreshold = outlier_threshold
        self._kernelWidth = kernel_width
        self._learningRate = learning_rate
        self._c = max_int_pdf
        self._selectionData = selection_size

    def _gmm(self, x_sel):
        self._m = np.size(x_sel, axis=1)
        print(
            "Gaussian Mixture Model for data representation, %.0f to %.0f samples"
            % (self._selectionData, self._mp)
        )
        gmm_n, gmm_m = np.shape(x_sel)
        if gmm_n > self._selectionData:
            x_sel = x_sel[1 : self._selectionData, :]
        self.gmm = skGMM(n_components=self._mp, covariance_type="diag").fit(
            x_sel
        )  # Set up SciKit GMM & train
        self._xc = self.gmm.means_  # Select means
        self._outlierMemory = self._xc  # Store in memory

    def update(self, xt):
        grad = self._kernel(self._xc, xt)  # Gradient
        if np.dot(self._alpha, grad) - 1 > 0:
            return self._xc, self._alpha, self._mp, np.size(self._outlierMemory, axis=0)
        self._alpha = self._alpha + self._learningRate * grad  # Gradient Descent step
        # if np.sum(self._alpha) >= self._c:  # If alpha's too high
        #     self._alpha = self._alpha / np.sum(self._alpha) * self._c  # Projecting for constraint (sum kernel < C)
        # Check whether GMM needs an update
        # todo implement own p here (That doesn't use the gmm's covariance ;)
        p = self.gmm.predict_proba(
            np.reshape(xt, (1, -1))
        )  # Calculate prob. xt belongs to xc
        if ~(p >= self._outlierThreshold).any():  # If P(xt ~in xc) < thres, save it
            self._outlierMemory = np.vstack((self._outlierMemory, xt))
            if np.size(self._outlierMemory, axis=0) >= self._memorySize:
                print("Refitting data representation, one moment s.v.p.")
                self._mp += 50
                self._memorySize += 100
                self.gmm = skGMM(
                    n_components=self._mp,
                    covariance_type="diag",
                    means_init=np.vstack((self._xc, np.zeros((50, self._m)))),
                )
                self.gmm.fit(self._outlierMemory)
                self._xc = np.zeros(self._mp)
                self._alpha = np.append(self._alpha, np.zeros(50))
                self._xc = self.gmm.means_
                self._outlierMemory = self._xc
        return self._xc, self._alpha, self._mp, np.size(self._outlierMemory, axis=0)

    def _kernel(self, x, y):
        if x.ndim == 1:
            return np.exp(
                -np.dot(x - y, x - y) / 2 / self._kernelWidth / self._kernelWidth
            )
        else:
            return np.exp(
                -np.diag(np.dot((x - y), (x - y).T))
                / 2
                / self._kernelWidth
                / self._kernelWidth
            )

    def visualize(self, plot_data, grid_size=None, cmap=None, levels=None):
        if grid_size is None:
            grid_size = 100
        if cmap is None:
            cmap = "RdBu_r"
        if levels is None:
            levels = 10
        xmi = 1.5 * np.min(self._xc)
        xma = 1.5 * np.max(self._xc)
        grid = np.linspace(xmi, xma, grid_size)
        f = np.zeros((grid_size, grid_size))
        for l in range(grid_size):
            for o in range(grid_size):
                for p, alph in enumerate(self._alpha):
                    f[o, l] = f[o, l] + alph * self._kernel(
                        np.array([grid[l], grid[o]]), self._xc[p, :]
                    )
        if self._m == 2:
            cont = plt.figure()
            plt.contour(grid, grid, f, levels=levels, linewidths=0.5, colors="k")
            plt.contourf(grid, grid, f, levels=levels, cmap=cmap)
            # ax1.colorbar(cntr1, ax=ax1)
            plt.scatter(plot_data[:, 0], plot_data[:, 1], c="k", s=1)
            plt.suptitle("Online Support Vector Clustering")
            plt.show()

    def predict(self, xp):
        f = np.dot(self._alpha, self._kernel(self._xc, xp))
        return f - 1
