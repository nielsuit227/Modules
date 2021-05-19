import gurobipy as grb
import numpy as np

"""
Simple Online Linear Support Vector Machine.

__init__:
 Only called upon startup.  -- LSVM(param)
 load_model should contain:
    reg :           regularization penalty
    eta :           learning rate
    w :             decision vector
    mean :          data mean (for normalization)
    variance :      data variance (for normalization)
 if load_model is unused you can feed the parameters manually. 

predict(x): 
 Simply predicts the class of data x.
 x should be an n x m matrix with n samples and m features

fit(x, y):
 Fits initial data. Uses an offline optimization. 
 If online optimization is desired from the start, use update(x, y) instead.
 x should be an n x m matrix with n samples and m features 
 
update(x, y):
 Updates the model for new data. 
 x should be an n x m matrix with n samples and m features
 
_update(x, y):
 Internal use for looped data. 
 
save_model():
 Creates a parameter file as required by load_model
"""


class LSVM(object):
    def __init__(
        self,
        load_model=None,
        regularization=1,
        learning_rate=0.001,
        weights=None,
    ):
        self._mu = []
        self._var = []
        if load_model is not None:
            self._c = load_model["reg"]
            self._eta = load_model["eta"]
            self._w = load_model["w"]
            self._mu = load_model["mean"]
            self._var = load_model["variance"]
        else:
            self._c = regularization
            self._eta = learning_rate
            if weights is not None:
                self._w = weights.reshape((len(weights), 1))
            else:
                self._w = []

    def predict(self, x):
        x = np.array(x)
        x -= self._mu
        x /= np.sqrt(self._var)
        return np.dot(x, self._w)

    def fit(self, x, y):
        # Changing data structure to Numpy
        x = np.array(x)
        y = np.array(y)
        n, m = np.shape(x)
        # Normalization
        self._mu = np.mean(x, axis=0)
        self._var = np.var(x, axis=0)
        x -= self._mu
        x /= np.sqrt(self._var)
        # Use Gurobi to optimize offline
        svm = grb.Model("LSVM")
        svm.setParam("OutputFlag", False)
        w = {}
        s = {}
        for i in range(m):
            w[i] = svm.addVar(vtype=grb.GRB.CONTINUOUS)
        for i in range(n):
            s[i] = svm.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=self._c)
            svm.addConstr(
                1 - y[i] * grb.quicksum(w[j] * x[i, j] for j in range(m)) <= s[i]
            )
        svm.setObjective(grb.quicksum(s[i] for i in range(n)))
        svm.optimize()
        self._w = [w[i].x for i in range(m)]

    def update(self, x, y):
        # Change data structure
        x = np.array(x)
        x -= self._mu
        x /= np.sqrt(self._var)
        if x.ndim != 1:
            for i in range(len(x)):
                self._update(x[i], y[i])
        else:
            self._update(x, y)

    def _update(self, x, y):
        # Update decision
        if y * np.dot(x, self._w) < 1:
            self._w = self._w + self._eta * y * x

    def save_model(self, filename):
        np.savez(
            filename + ".npz",
            w=self._w,
            reg=self._c,
            eta=self._eta,
            mean=self._mu,
            variance=self._var,
        )
