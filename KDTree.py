# Python 3.6.5.
# Numpy 1.16.0.
# Matplotlib 3.0.2.
import matplotlib.pyplot as plt
import numpy as np


def sortnode(val):
    return val[2]


# todo: implement multiple splitting algo's (now median)
# todo: make predict function? for buckets
# todo: make insert function
# todo: make delete function
"""
Builds a K-dimensional tree.
Functions with inputs & outputs:

__init__  Initializes the model.

          leafsize (int):         Determines maximum datapoints in leaf.
          kdim (int):             Amount of dimensions over which the tree is separated
          random (bool):          Whether or not dimension is chosen random.
          pc_sort (bool):         Whether or not the algo is projected on its principle components
          sort_algo:              Sorting algorithm {'quicksort','mergesort','heapsort','stable'}
          savedata (bool):        Whether the tree saves all data or just the indices

Output:   Model of tree

fit       Splits the dataset into branches until each branch contains less datapoints
          than the specified leaf size.

          data (n x m)            Data for tree (n samples, m features)

Output:   Actual Tree as list. [(Data, indices, node, split dimension, split value)]

plot      Plots tree (in 2D only)

          Self

leaf_dist     Plots a boxplot of the average distance within a leaf. Can be of great
              help to determine radius size for Radius Nearest Neighbor.

              Self

exact_r_nn    Calculates the exact Radius Nearest Neighbors.

              eps:                Radius for radius nearest neighbors
              point:              Startpoint for NN

bin_r_nn      Calculates the Radius Nearest Neighbors within the same leaf.

              eps:                Radius for radius nearest neighbors
              point:              Startpoint for NN
"""


class KDTree(object):
    def __init__(
        self,
        leafsize=10,
        kdim=None,
        random=False,
        pc_sort=False,
        sort_algo="quicksort",
        savedata=True,
    ):
        if sort_algo not in {"quicksort", "mergesort", "heapsort", "stable"}:
            raise ValueError("Select sorting algo of numpy.argsort v1.16.0.")
        # Pass settings
        self._leafs = leafsize
        self._rk = random
        self._kdim = kdim
        self._sort_algo = sort_algo
        self._sort = pc_sort
        self._sd = savedata
        # Generate empties
        self._data = []
        self._n = []
        self._m = []
        self._tree = []
        self._stack = []
        self._sdim = []

    def fit(self, data):
        self._data = data.copy()
        self._n, self._m = np.shape(self._data)
        if self._kdim is None:
            self._kdim = self._m
        if self._sort:
            u, s, v = np.linalg.svd(self._data, full_matrices=False, compute_uv=True)
            self._data = np.dot(self._data, v)
        if self._rk:
            self._sdim = np.arange(self._kdim)
            ind = self._sdim[np.random.randint(0, self._kdim)]
        else:
            ind = 0
        idx = np.argsort(data[:, ind], kind=self._sort_algo)
        data = data[idx, :]
        # Tree: data, data index, node, splitdim, splitval
        self._tree = [(None, idx, 0, ind, data[int(self._n / 2), ind])]
        # Stack: data, data index, node, leaf, depth
        self._stack.append(
            (
                data[: int(self._n / 2), :],
                idx[: int(self._n / 2)],
                1,
                int(self._n / 2) < self._leafs,
                0,
            )
        )
        self._stack.append(
            (
                data[int(self._n / 2) :, :],
                idx[int(self._n / 2) :],
                2,
                int(self._n / 2) < self._leafs,
                0,
            )
        )
        while self._stack:
            data, idx, node, leaf, depth = self._stack.pop()
            if leaf:
                if self._sd:
                    self._tree.append((data, idx, node, None, None))
                else:
                    self._tree.append((1, idx, node, None, None))
            else:
                if self._rk:
                    ind = self._sdim[np.random.randint(0, self._kdim)]
                else:
                    ind = np.remainder(depth + 1, self._m)
                tidx = np.argsort(data[:, ind], kind=self._sort_algo)
                data = data[tidx, :]
                idx = idx[tidx]
                ndata, ndim = np.shape(data)
                self._stack.append(
                    (
                        data[: int(ndata / 2), :],
                        idx[: int(ndata / 2)],
                        node * 2 + 1,
                        int(ndata / 2) < self._leafs,
                        depth + 1,
                    )
                )
                self._stack.append(
                    (
                        data[int(ndata / 2) :, :],
                        idx[int(ndata / 2) :],
                        node * 2 + 2,
                        int(ndata / 2) < self._leafs,
                        depth + 1,
                    )
                )
                self._tree.append((None, idx, node, ind, data[int(ndata / 2), ind]))
        self._tree.sort(key=sortnode)
        return self._tree

    def exact_r_nn(self, eps, point, maxeval=None):
        i = 0
        tidx = []
        stack = [self._tree[0]]
        while stack:
            data, idx, node, sdim, sval = stack.pop()
            if data is not None:
                i += 1
                distance = np.sqrt(np.sum((self._data[idx, :] - point) ** 2, 1))
                tidx = np.unique(np.hstack((tidx, idx[np.where(distance < eps)])))
                # print('Exact Radius Nearest Neighbor expanded to %.0f samples' % np.size(tidx))
                if maxeval is not None:
                    if i >= maxeval:
                        return tidx.astype(int)
            else:
                if point[sdim] > sval - eps:
                    stack.append(self._tree[node * 2 + 2])
                if point[sdim] < sval + eps:
                    stack.append(self._tree[node * 2 + 1])
        return tidx.astype(int)

    def bin_r_nn(self, eps, point):
        stack = [self._tree[0]]
        while stack:
            data, idx, node, sdim, sval = stack.pop()
            if data is not None:
                distance = np.sqrt(np.sum((self._data[idx, :] - point) ** 2, 1))
                return idx[np.where(distance < eps)]
            else:
                if point[sdim] > sval:
                    stack.append(self._tree[node * 2 + 2])
                else:
                    stack.append(self._tree[node * 2 + 1])

    def plot(self):
        plt.figure()
        xmi = min(1.5 * np.min(self._data[:, 0]), 0.5 * np.min(self._data[:, 0]))
        xma = 1.5 * np.max(self._data[:, 0])
        ymi = min(1.5 * np.min(self._data[:, 1]), 0.5 * np.min(self._data[:, 1]))
        yma = 1.5 * np.max(self._data[:, 1])
        hrect = [(0, xmi, xma, ymi, yma)]
        while hrect:
            parent, pxmi, pxma, pymi, pyma = hrect.pop()
            data, idx, node, sdim, sval = self._tree[parent]
            if data is not None:
                plt.scatter(self._data[idx, 0], self._data[idx, 1], s=4)
                continue
            if sdim == 1:
                hrect.append((2 * parent + 2, pxmi, pxma, sval, pyma))
                hrect.append((2 * parent + 1, pxmi, pxma, pymi, sval))
                plt.plot([pxmi, pxma], [sval, sval])
            else:
                hrect.append((2 * parent + 1, pxmi, sval, pymi, pyma))
                hrect.append((2 * parent + 2, sval, pxma, pymi, pyma))
                plt.plot([sval, sval], [pymi, pyma])

    def leaf_dist(self):
        Z = []
        Y = []
        for branch in self._tree:
            if branch[0] is not None:
                if branch[0] == 1:
                    data = self._data[branch[1], :]
                else:
                    data = branch[0]
                leafSize = np.size(branch[1])
                dist = np.zeros(leafSize)
                for i in range(leafSize):
                    dist[i] = np.mean(np.sum((data - data[i, :]) ** 2, 1))
                Z.append(np.mean(dist))
                Y.append(dist)
        plt.boxplot(Z)
        plt.suptitle("Boxplot of average distance within leaf")
        plt.figure()
        plt.boxplot(Y)
        plt.suptitle("Boxplot of all distances within leaf")
