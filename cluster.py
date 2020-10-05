import numpy as np
from sklearn.mixture import GaussianMixture as skGMM
'''
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
'''
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
        seeds = np.zeros(self._n, dtype='bool')
        if np.size(point) == 1:
            self._checked[point] = True
            if not self._tree:
                seedsindex = self.bruteforce(self._data[point, :])
            else:
                seedsindex = self._tree.exact_r_nn(self._eps, self._data[point, :], maxeval=10)
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

    def predict_outlier(self, point):
        seeds = self.neighbor(point)
        if np.sum(seeds) > self._minpoints:
            return False
        else:
            return True
        

class OSVC(object):

    def __init__(self, multipliers=100, kernel_width=2.0, learning_rate=0.001, max_int_pdf=100.0, memory_size=5000,
                 outlier_threshold=0.2, selection_size=5000, ac=None, xc=None):
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
        print('Gaussian Mixture Model for data representation, %.0f to %.0f samples' % (self._selectionData, self._mp))
        gmm_n, gmm_m = np.shape(x_sel)
        if gmm_n > self._selectionData:
            x_sel = x_sel[1:self._selectionData, :]
        self.gmm = skGMM(n_components=self._mp, covariance_type='diag').fit(x_sel)  # Set up SciKit GMM & train
        self._xc = self.gmm.means_          # Select means
        self._outlierMemory = self._xc      # Store in memory

    def update(self, xt):
        grad = self._kernel(self._xc, xt)  # Gradient
        if np.dot(self._alpha, grad) - 1 > 0:
            return self._xc, self._alpha, self._mp, np.size(self._outlierMemory, axis=0)
        self._alpha = self._alpha + self._learningRate * grad  # Gradient Descent step
        # if np.sum(self._alpha) >= self._c:  # If alpha's too high
        #     self._alpha = self._alpha / np.sum(self._alpha) * self._c  # Projecting for constraint (sum kernel < C)
        # Check whether GMM needs an update
        # todo implement own p here (That doesn't use the gmm's covariance ;)
        p = self.gmm.predict_proba(np.reshape(xt, (1, -1)))  # Calculate prob. xt belongs to xc
        if ~(p >= self._outlierThreshold).any():  # If P(xt ~in xc) < thres, save it
            self._outlierMemory = np.vstack((self._outlierMemory, xt))
            if np.size(self._outlierMemory, axis=0) >= self._memorySize:
                print('Refitting data representation, one moment s.v.p.')
                self._mp += 50
                self._memorySize += 100
                self.gmm = skGMM(n_components=self._mp, covariance_type='diag',
                                 means_init=np.vstack((self._xc, np.zeros((50, self._m)))))
                self.gmm.fit(self._outlierMemory)
                self._xc = np.zeros(self._mp)
                self._alpha = np.append(self._alpha, np.zeros(50))
                self._xc = self.gmm.means_
                self._outlierMemory = self._xc
        return self._xc, self._alpha, self._mp, np.size(self._outlierMemory, axis=0)

    def _kernel(self, x, y):
        if x.ndim == 1:
            return np.exp(-np.dot(x - y, x - y) / 2 / self._kernelWidth / self._kernelWidth)
        else:
            return np.exp(-np.diag(np.dot((x - y), (x - y).T)) / 2 / self._kernelWidth / self._kernelWidth)

    def visualize(self, plot_data, grid_size=None, cmap=None, levels=None):
        if grid_size is None:
            grid_size = 100
        if cmap is None:
            cmap = 'RdBu_r'
        if levels is None:
            levels = 10
        xmi = 1.5 * np.min(self._xc)
        xma = 1.5 * np.max(self._xc)
        grid = np.linspace(xmi, xma, grid_size)
        f = np.zeros((grid_size, grid_size))
        for l in range(grid_size):
            for o in range(grid_size):
                for p, alph in enumerate(self._alpha):
                    f[o, l] = f[o, l] + alph * self._kernel(np.array([grid[l], grid[o]]), self._xc[p, :])
        if self._m == 2:
            cont = plt.figure()
            plt.contour(grid, grid, f, levels=levels, linewidths=0.5, colors='k')
            plt.contourf(grid, grid, f, levels=levels, cmap=cmap)
            # ax1.colorbar(cntr1, ax=ax1)
            plt.scatter(plot_data[:, 0], plot_data[:, 1], c='k', s=1)
            plt.suptitle('Online Support Vector Clustering')
            plt.show()

    def predict(self, xp):
        f = np.dot(self._alpha, self._kernel(self._xc, xp))
        return f-1



