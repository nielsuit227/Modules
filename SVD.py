import numpy as np
from scipy import linalg


# Not working...
def rsvdpi(a, k, p, s):
    n, m = np.shape(a)
    O = np.random.randn(m, k + s)
    Q = np.dot(a, O)
    for i in range(p+1):
        if i < p:
            P, L, U = linalg.lu(Q)
            Q = np.dot(P.T, L)
        else:
            Q = eigsvd(Q)[0]
        Q = np.dot(a, np.dot(a.T, Q))
    b = np.dot(Q.T, a)
    V, S, U = eigsvd(b.T)
    ind = np.arange(s, k+s)
    return np.dot(Q, U[:, ind]), np.diag(S)[ind, ind], V[:, ind]


def rsvd(a, k, p, s):
    n, m = np.shape(a)
    O = np.random.randn(m, k + s)
    Q = linalg.orth(np.dot(a, O))
    for i in range(p):
        G = linalg.orth(np.dot(a.T, Q))
        Q = linalg.orth(np.dot(a, G))
    b = np.dot(Q.T, a)
    u, s, v = eigsvd(b)
    u = np.dot(Q, u)
    return u[:, :k], s[:k], v[:, :k]


def eigsvd(a):
    transposeflag = None
    if np.size(a, axis=0) < np.size(a, axis=1):
        transposeflag = True
        a = a.T
    b = np.dot(a.T, a)
    eval, v = np.linalg.eig(b)
    v = v[:, np.argsort(-eval)]
    eval = eval[np.argsort(-eval)]
    s = np.sqrt(eval)
    u = np.dot(np.dot(a, v), np.diag(1/s))
    if transposeflag:
        u = v.T
        v = u.T
    return u, s, v