# Version control
# Python 3.6.5
# Numpy 1.16.0
import numpy as np

'''
Online Distributionally Robust Support Vector Machine. 

__init__:
 Only called upon startup -- ODRSVM(param)
 load_model should contain:
    multipliers :   Optimized Lagrangian multipliers (weights)
    oca_weights :   Data representation weights
    oca_labels :    Data representation classes/labels
    oca_centers :   Data representation features/centers
    kernel_width :  Kernel width used for Gaussian Kernel
    ambiguity :     Optimized ambiguity for each oca_centers
    mean :          Data mean (used for normalization)
    variance :      Data variance (used for normalization)
    reg :           Regularization penalty
    was_norm :      Wasserstein norm (ambiguity)
    was_rad :       Wasserstein radius (ambiguity)
    oca_rad :       Radius for oca
    eta :           Learning rate
 if load_model is unused you can feed everything manually
 
_kernel(x, y):
 Internal use, evaluates kernel function at feature x and feature y. 
 
_gram(x):
 Internal use, returns gram matrix for data matrix x (row matrix of data samples)
 
_oca(x, y):
 Internal use, updates online covering algorithm for new data point x, y. 
 
update(x, y):
 Updates the online covering algorithm and current decision vector for feature x and label y.
 x should be a n x m matrix with n samples and m features. 
 
_update(x, y):
 Internal use, solely for looped data.
 
fit(x, y):
 Initial optimization. Online so simply calls update(x, y).
 x should be a n x m matrix with n samples and m features.  
 
save_model():
 Stores all required model parameters which can directly be used next initialization. 
'''


class ODRSVM(object):

    def __init__(self,
                 oca_rad=0,
                 regularization=1,
                 wasserstein_rad=0,
                 wasserstein_norm=2,
                 learningrate=0.1,
                 kernelwidth=1,
                 shuffle=False,
                 normalization=False,
                 load_model=None
                 ):
        # Pass parameters
        self._normalization = normalization
        self._c = regularization
        self._rad = oca_rad
        self._eps = wasserstein_rad
        self._norm = wasserstein_norm
        self._eta = learningrate
        self._sig = kernelwidth
        self._shuffle = shuffle
        # Initialize so PyCharm doesn't whine
        self._oca_l = []  # Online Covering Algorithm - Labels
        self._oca_w = []  # Online Covering Algorithm - Weights
        self._oca_c = []  # Online Covering Algorithm - Centers
        self._x_window = []  # Memory for class window
        self._y_window = []  # Memory for label window
        self._k = 1  # Tracker of data set size
        self._m = []  # Feature dimension
        self._alpha = np.array([[0]], dtype='float')  # Decision variable (Lagrangian multipliers)
        self._agrad = np.array([[0]], dtype='float')  # and its derivative
        self._y = []  # Ambiguity set
        self._kt = []  # Kernel vector
        self._mu = []  # Data mean
        self._var = []  # Data variance
        if load_model is not None:
            self._alpha = load_model['multipliers']
            self._oca_w = load_model['oca_weights']
            self._oca_l = load_model['oca_labels']
            self._oca_c = load_model['oca_centers']
            self._sig = load_model['kernel_width']
            self._y = load_model['ambiguity']
            self._k = len(self._alpha)
            self._mu = load_model['mean']
            self._var = load_model['variance']
            self._norm = load_model['was_norm']
            self._eps = load_model['was_rad']
            self._eta = load_model['eta']
            self._c = load_model['kernel_width']
            self._rad = load_model['oca_rad']

    def _kernel(self, x, y):
        if x.ndim == 1:
            return np.exp(-np.dot(x - y, x - y) / 2 / self._sig ** 2)
        elif x.ndim > 1 and y.ndim == 1:
            return np.exp(-np.diag(np.dot((x - y), (x - y).T)) / 2 / self._sig ** 2).reshape((self._k, 1))
        elif x.ndim > 1 and y.ndim > 1:
            g = np.zeros((np.shape(x)[0], np.shape(y)[0]))
            for i, v1 in enumerate(x):
                for j, v2 in enumerate(y):
                    g[i, j] = self._kernel(v1, v2)
            return g

    def _gram(self, x):
        n, m = np.shape(x)
        g = np.zeros((n, n))
        for q in range(n):
            g[q] = self._kernel(x, x[q]).reshape(self._k)
        return g

    def predict(self, x):
        x = np.array(x)
        if self._normalization is True:
            x -= self._mu
            x /= np.sqrt(self._var)
        if x.ndim == 1:
            return np.sum(self._alpha * self._oca_w * self._oca_l * self._kernel(self._oca_c - self._y /
                                                                                 self._oca_w, x))
        else:
            n, m = np.shape(x)
            yp = np.zeros(n)
            for i, xt in enumerate(x):
                print(" [ %.2f %% ]" % ((i+1)*100/len(yp)), end="\r")
                yp[i] = np.sum(self._alpha * self._oca_w * self._oca_l * self._kernel(self._oca_c - self._y /
                                                                                      self._oca_w, xt))
            return yp

    def _oca(self, x, y):
        if len(self._oca_c) == 0:
            self._m = len(x)
            self._oca_c = x.reshape((1, self._m))
            self._x_window = x.reshape((1, self._m))
            self._y_window = y.reshape((1, 1))
            self._oca_l = y.reshape((1, 1))
            self._oca_w = np.array([[1]], dtype='float')
            self._y = np.zeros_like(self._oca_c)
            return
        elif y == 1:
            ind = np.where(self._oca_l == 1)[0]
            if len(self._oca_c[ind]) != 0:
                dist = np.sqrt(np.sum((self._oca_c[ind] - x) ** 2, axis=1))
                if (dist < self._rad).any():
                    self._x_window = np.vstack((self._x_window, x))
                    self._y_window = np.vstack((self._y_window, y))
                    self._oca_w[ind[dist < self._rad]] += 1 / np.sum(dist < self._rad)
                    return
        elif y == -1:
            ind = np.where(self._oca_l == -1)[0]
            if len(self._oca_c[ind]) != 0:
                dist = np.sqrt(np.sum((self._oca_c[ind] - x) ** 2, axis=1))
                if (dist < self._rad).any():
                    self._x_window = np.vstack((self._x_window, x))
                    self._y_window = np.vstack((self._y_window, y))
                    self._oca_w[ind[dist < self._rad]] += 1 / np.sum(dist < self._rad)
                    return
        self._x_window = np.vstack((self._x_window, x))
        self._y_window = np.vstack((self._y_window, y))
        self._oca_c = np.vstack((self._oca_c, x))
        self._oca_l = np.vstack((self._oca_l, y))
        self._oca_w = np.vstack((self._oca_w, 1))
        self._y = np.vstack((self._y, np.zeros((len(self._oca_c) - self._k, self._m))))
        self._alpha = np.vstack((self._alpha, np.zeros((len(self._oca_c) - self._k, 1))))
        self._agrad = np.vstack((self._agrad, np.zeros((len(self._oca_c) - self._k, 1))))
        self._k = len(self._oca_c)
        return

    def update(self, x, y):
        # Change data structure to Numpy
        x = np.array(x)
        y = np.array(y)
        if self._normalization is True:
            x -= self._mu
            x /= np.sqrt(self._var)
        
        # Check whether matrix or vector
        if x.ndim != 1:
            for i in range(len(x)):
                print('       [ %.0f %%]' % (i * 100 / len(y)), end='\r')
                self._update(x[i], y[i])
        else:
            self._update(x, y)
            
    def _update(self, x, y):
        # Online covering algorithm
        self._oca(x, y)
        self._kt = self._kernel(self._oca_c - self._y / self._oca_w, x)

        # Ambiguity update (Smoothed, Hazan 2017)
        window = 20
        tolerance = 0.01
        converged = False
        self._x_window = self._x_window[-window:]
        self._y_window = self._y_window[-window:]
        if self._eps != 0:
            # while converged is False:
            kt = self._kernel(self._oca_c - self._y / self._oca_w, self._x_window)
            kt *= (np.outer(self._oca_l, self._y_window) == -1)*1
            kt_3d = np.ones((self._m, self._k, len(self._x_window))) * kt
            mem_3d = np.ones((self._m, self._k, len(self._x_window))) * self._x_window[:, None, :].T
            x_3d = np.ones((self._m, self._k, len(self._x_window))) * self._oca_c.T[:, :, None]
            wu_3d = np.ones((self._m, self._k, len(self._x_window))) * self._y.T[:, :, None] / self._oca_w.T[:, :, None]
            grad = -self._eta * self._alpha * np.sum(kt_3d * (mem_3d - x_3d + wu_3d), axis=2).T
            norm = np.sum(np.linalg.norm(self._y + grad, self._norm, axis=1))
            if norm > self._eps * self._k:
                grad = (self._y + grad) / norm * self._k * self._eps - self._y
            self._y += grad
                # if np.sum(np.linalg.norm(grad, self._norm, axis=1)) < tolerance * self._k:
                #     converged = True

        # Decision update
        # self._kt = self._kernel(self._oca_c - self._y / self._oca_w, x)
        if y * np.sum(self._oca_w * self._kt * self._alpha * self._oca_l) < 1:
            self._agrad -= y * self._oca_w * self._oca_l * self._kt
            self._alpha = np.minimum(np.exp(-self._eta * self._agrad - 1), self._c)
            if self._oca_l.ndim != 0:           # Moet hier niet een thetaatje bij?
                sa = np.sum(self._alpha * self._oca_w)
                self._alpha[self._oca_l == 1] = self._alpha[self._oca_l == 1] * self._oca_w[self._oca_l == 1] * sa / 2 \
                                        / np.sum(self._alpha[self._oca_l == 1] * self._oca_w[self._oca_l == 1])
                self._alpha[self._oca_l == -1] = self._alpha[self._oca_l == -1] * self._oca_w[self._oca_l == -1] * sa / 2 \
                                        / np.sum(self._alpha[self._oca_l == -1] * self._oca_w[self._oca_l == -1])

    def fit(self, x, y):
        # Change data structure
        x = np.array(x)
        y = np.array(y)
        # Normalization
        if self._normalization:
            self._mu = np.mean(x, axis=0)
            self._var = np.var(x, axis=0)
            x -= self._mu
            x /= np.sqrt(self._var)
        # Shuffle data
        if self._shuffle:
            ind = np.random.permutation(len(x))
            x = x[ind]
            y = y[ind]
        # Optimize decision
        for i in range(len(y)):
            print('       [ %.0f %%]' % (i * 100 / len(y)), end='\r')
            self._update(x[i], y[i])

    def save_model(self):
        np.savez('optimized_odrsvm.npz',
                 multipliers=self._alpha,
                 oca_weights=self._oca_w,
                 oca_labels=self._oca_l,
                 oca_centers=self._oca_c,
                 kernel_width=self._sig,
                 ambiguity=self._y,
                 mean=self._mu,
                 variance=self._var,
                 waas_norm=self._norm,
                 was_rad=self._eps,
                 eta=self._eta,
                 oca_rad=self._rad)
