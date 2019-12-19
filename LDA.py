import numpy as np
from scipy import linalg
from self.SVD import rsvd


# Linear Discriminant Analysis.
# Can be used for linear classification or supervised dimension reduction.
# Assumes same class covariances.
# Just binary implementation.
# Saves ordered transformation matrix.
# [1] : https://sebastianraschka.com/Articles/2014_python_lda.html
# [2] : Making Fisher Discriminant Analysis Scalable - Tu et al. - 2014
# [3] : Training LDA in Linear Time - Cai et al. - 2008


class LDA(object):
    """
    Linear Discriminant Analysis with different solvers (also approximate solvers)
    for binary classification. Unfortunatly only 'eig' and 'twostage' solvers are
    not restricted in output dimension size.
    """

    def __init__(self, n_components=None, method='qrsvd'):
        self._od = n_components
        self._scalings = []
        self._transform_matrix = []
        self._x = []
        self._y = []
        self._n = []
        self._m = []
        self._mu_p = []
        self._mu_n = []
        self._mu = []
        self._n_p = []
        self._n_n = []
        self._x = []
        self._method = method

    def fit(self, x, y):
        if set(y) != {1, -1}:
            raise ValueError('Only binary classes {+1,-1} are accepted.')
        self._n = len(y)
        self._m = np.size(x, axis=1)
        self._n_n = np.sum(y == 1)
        self._n_p = np.sum(y == -1)
        self._mu = np.mean(x, axis=0)
        self._mu_p = np.mean(x[y == 1, :], axis=0)
        self._mu_n = np.mean(x[y == -1, :], axis=0)
        self._x = x
        self._y = y
        if self._method == 'twostage':
            return self.fit_twostage(x, y)
        elif self._method == 'eig':
            return self.fit_eig(x, y)
        elif self._method == 'qrsvd':
            return self.fit_qr(x, y)
        elif self._method == 'svd':
            return self.fit_svd(x, y)
        elif self._method == 'srda':
            return self.fit_srda(x, y)
        else:
            raise NameError('LDA solver not implemented, chose either "eig", "svd", "twostage" or "qrsvd"')

    # Classical LDA             [1]
    def fit_eig(self, x, y):
        # Class variances
        s_w = np.dot((x[y == -1, :] - self._mu_n).T, (x[y == -1, :] - self._mu_n))/self._n
        s_w += np.dot((x[y == 1, :] - self._mu_p).T, (x[y == 1, :] - self._mu_p))/self._n
        # Within class variances
        s_b = self._n_n*np.outer((self._mu_n - self._mu), (self._mu_n - self._mu))/self._n
        s_b += self._n_p*np.outer((self._mu_p - self._mu), (self._mu_p - self._mu))/self._n
        # Total variance
        # s_t = np.dot((x-self._mu).T, (x-self._mu))/self._n
        # Outputs
        self._scalings, self._transform_matrix = np.linalg.eig(np.dot(np.linalg.inv(s_w), s_b))
        return self._scalings, self._transform_matrix

    # SVD based LDA             [2]
    def fit_svd(self, x, y):
        # x = (x-self._mu)/np.sqrt(np.var(x, axis=0))
        hb = 1 / np.sqrt(self._n) * np.reshape((np.sqrt(self._n_n) * (self._mu_n - self._mu),
                                                np.sqrt(self._n_p) * (self._mu_p - self._mu)), (2, self._m)).T
        ht = 1 / np.sqrt(self._n) * (x - self._mu).T
        u, s, v = np.linalg.svd(ht)
        b = np.dot(np.dot(np.diag(1/s), u.T), hb)
        p, q, r = np.linalg.svd(b, full_matrices=False, compute_uv=True)
        self._transform_matrix = np.dot(u, np.dot(np.diag(1/s), p))
        self._scalings = q
        return self._scalings, self._transform_matrix

    # Two stage PCA/LDA algo    [3]
    # PCA first with SVD, LDA according to eigenvalue decomposition.
    def fit_twostage(self, x, y):
        wp = 1/self._n_p*np.ones((self._n_p, self._n_p))
        wn = 1/self._n_n*np.ones((self._n_n, self._n_n))
        w = linalg.block_diag(wp, wn)
        xb = x.copy()
        xb[y == 1, :] = xb[y == 1, :] - self._mu_p
        xb[y == -1, :] = xb[y == -1, :] - self._mu_n
        u, s, v = np.linalg.svd(np.array(xb).T, full_matrices=False, compute_uv=True)
        # Think I can do this too with the double SVD from [Tu et al. 14]
        self._scalings, b = np.linalg.eig(np.dot(v, np.dot(w, v.T)))
        a = np.dot(u, np.dot(np.diag(1/s), b))
        self._transform_matrix = a/np.linalg.norm(a, axis=1)
        return self._scalings, self._transform_matrix

    # QR-LDA       [2]
    # todo implement randomized SVD for time complexity
    def fit_qr(self, x, y):
        # Precursors of scatter matrices
        hb = 1 / np.sqrt(self._n) * np.reshape((np.sqrt(self._n_n) * (self._mu_n - self._mu),
                                               np.sqrt(self._n_p) * (self._mu_p - self._mu)), (2, self._m)).T
        ht = 1/np.sqrt(np.size(x, axis=0))*(x-self._mu).T
        q = np.linalg.matrix_rank(hb)
        u, s, v = np.linalg.svd(ht, full_matrices=False, compute_uv=True)
        z1 = np.reshape(u[:, 0], (self._m, 1))
        Q, R, P = linalg.qr(hb-np.dot(np.dot(z1, z1.T), hb), pivoting=True)
        z2 = Q[:, :(self._m-1)]
        z = np.hstack((z1, z2))
        ahb = np.dot(z.T, hb)
        aht = np.dot(z.T, ht)
        # Original svm LDA
        u, s, v = np.linalg.svd(aht, full_matrices=False, compute_uv=True)
        b = np.dot(np.dot(np.diag(1/s), u.T), ahb)
        P, Q, R = np.linalg.svd(b, full_matrices=False, compute_uv=True)
        self._transform_matrix = np.dot(z, np.dot(np.dot(u, np.diag(1/s)), P))
        self._scalings = Q
        return self._scalings, self._transform_matrix

    # Spectral Regression Discriminant Analysis  [3]
    def fit_srda(self, x, y):
        e0 = np.ones(self._n)
        y1 = np.zeros_like(e0)
        y2 = np.zeros_like(e0)
        y1[:self._n_p] = 1/self._n_p
        y2[self._n_p:] = 1/self._n_n
        e1 = y1 - np.dot(y1, e0)/np.dot(e0, e0)*e0
        e2 = y2 - np.dot(y2, e0)/np.dot(e0, e0)*e0 - np.dot(y2, e1)/np.dot(e1, e1)*e1
        xb = x.copy()
        xb[y == 1, :] -= self._mu_p
        xb[y == -1, :] -= self._mu_n
        a = np.dot(np.dot(np.linalg.inv((np.dot(xb.T, xb))), xb.T), np.array([e0, e1, e2]).T[:, :self._od])
        self._transform_matrix = a/np.linalg.norm(a)
        return self._transform_matrix

    def transform(self, x=None, y=None, n_components=None):
        if x is None:
            x = self._x
        if n_components is not None:
            self._od = n_components
        if not self._od:
            raise ValueError('Output dimension undefined, '
                             'please specify n_components in either the model or the transform')
        if not self._transform_matrix:
            self.fit(x, y)
        return np.dot(x, self._transform_matrix[:, :self._od])






