import matplotlib.pyplot as plt
import numpy as np

"""
Density Bases Spatial Clustering for Applications with Noise. [1]
Clustering algorithm parameterized by the cluster radius and the minimum points in this radius to be defined a cluster.
The algorithm efficiently checks for each point how many points are within said radius. If this exceeds the points
defined as minpoints it is considered a cluster center and all adjacent points are considered to be in the cluster.
This process is signficantly sped up by using a Decision Tree and Radius Nearest Neigbhors.
Functions with inputs:

__init__ (radius, minpoints, tree)
Initializes the model.

radius          Distance defining whether points are within clusters
minpoints:      Minimum points within radius to be defined cluster
tree:           (Optional) Will exploit data structure of Radius Nearest Neighbor to speed up the algorithm.
 

Fit(data)
Runs the DBSCAN algorithm for the given data.

data:           n x m dataset with n samples containing m features. Normalized data is highly recommended as
                the algorithm uses a distance based clustering metric.

:returns        cluster ID's, -1 is reserved for outliers/noise.

expandcluster(point)
*** internal use ***
Function called if a point is a cluster center and checks all points within the cluster.

neighbor(point)
*** internal use ***
Returns a boolean with datapoints that are within cluster range. Might either use 'bruteforce' or a radius nearest
neigbhor algorithm from a defined tree. Note that in this case the exact_r_nn is used, this can also be an approximated
radius nearest neighbor to further speed up the algo (bin_r_nn).

nextid(seeds)
*** internal use ***
Seeds is a boolean vector containing indices of those that need a cluster ID assigned.
This function simply increments the cluster ID and assigns them.

bruteforce(point)
*** internal use ***
returns seeds which are within Euclidean distance as defined by radius.



[1] A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise - Ester et al. - 1996
"""
# todo implement merge
class DBSCAN(object):
    def __init__(self, radius=1.0, minpoints=50, tree=None):

        # Pass variables
        if tree is not None:
            self._tree = tree
        else:
            self._tree = []
        self._eps = radius
        self._minpoints = minpoints
        self._next_cluster_id = 0
        # Create empties
        self._n = []
        self._m = []
        self._data = []
        self._clusterid = []
        self._checked = []
        self._outliers = []

    def fit(self, data):
        self._data = data
        self._n, self._m = np.shape(self._data)
        self._clusterid = -1 * np.ones(self._n)
        self._checked = False * np.ones(self._n)
        for i in range(self._n):
            if self._clusterid[i] == -1:
                # print('[%.2f %%] Last Cluster Size: %.0f'
                #       % (100*i/self._n, np.sum(self._clusterid == np.max(self._clusterid))))
                self.expandcluster(i)
        return self._clusterid

    # _clusterid is initialized as -1 (outliers) and keeps track of assigned clusters.
    # _checked is initalized as False and every point is marked True when it is checked against other points

    def expandcluster(self, point):
        seeds = self.neighbor(point)
        if np.sum(seeds) < self._minpoints:
            return self._clusterid
        else:
            self._clusterid[seeds] = self.nextid(seeds)
            seeds[np.where(seeds)[0][0]] = False
            while np.sum(seeds) != 0:
                npoint = np.where(seeds)[0][0]
                nseeds = self.neighbor(npoint)
                if np.sum(nseeds) >= self._minpoints:
                    self._clusterid[nseeds] = self._clusterid[npoint]
                    seeds = np.maximum(nseeds, seeds)
                    seeds = np.logical_and(seeds == True, self._checked == False)
                else:
                    seeds[npoint] = False
        return

    # seeds is a vector which marks unchecked data samples in the neighborhood of the cluster. Points that are radius
    # of extended clusters are added. It's a to do list for checking distances pretty much.

    def neighbor(self, point):
        seeds = np.zeros(self._n, dtype="bool")
        if np.size(point) == 1:
            self._checked[point] = True
            if not self._tree:
                seedsindex = self.bruteforce(self._data[point, :])
            else:
                seedsindex = self._tree.exact_r_nn(
                    self._eps, self._data[point, :], maxeval=10
                )
        else:
            if not self._tree:
                seedsindex = self.bruteforce(point)
            else:
                seedsindex = self._tree.exact_r_nn(self._eps, point, maxeval=10)
        seeds[seedsindex] = True
        return seeds

    def nextid(self, seeds):
        self._next_cluster_id += 1
        return np.ones(int(np.sum(seeds))) * self._next_cluster_id

    def bruteforce(self, point):
        distance = np.sqrt(np.sum((self._data - point) ** 2, 1))
        return np.where(distance < self._eps)

    def plot(self):
        plt.figure()
        self._outliers = self._clusterid == -1
        plt.scatter(self._data[:, 0], self._data[:, 1], c="g", s=1)
        plt.scatter(
            self._data[self._outliers, 0], self._data[self._outliers, 1], c="r", s=1
        )
        plt.suptitle("Estimated clusters")
        plt.legend(self._outliers)

    def predict_outlier(self, point):
        seeds = self.neighbor(point)
        if np.sum(seeds) > self._minpoints:
            return False
        else:
            return True
