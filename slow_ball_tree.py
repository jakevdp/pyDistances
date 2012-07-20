"""
Pure Python BallTree
--------------------

This is a pure python ball tree.  It is slow, but should give correct
results, and is much more readable than the cython version of the
ball tree code.

For consistency, it uses the distance metric framework to enable
use of arbitrary distance metrics.
"""
import numpy as np
from distmetrics import DistanceMetric
from ball_tree import BallTree

class SlowBallTree:
    def __init__(self, X, leaf_size=20, metric='euclidean', **kwargs):
        self.X = np.asarray(X)
        self.leaf_size = leaf_size
        self.dm = DistanceMetric(metric, **kwargs)

        # build the tree
        self.indices = np.arange(X.shape[0], dtype=int)
        self.head_node = Node(self.X, self.indices,
                              self.leaf_size, self.dm)

    def query(self, X, k):
        # check k
        if k > self.X.shape:
            raise ValueError("k must be less than the "
                             "number of points in the tree")

        # check X and resize if necessary
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        N = X.shape[0]
        
        if X.ndim != 2 or X.shape[1] != self.X.shape[1]:
            raise ValueError("last dimension of X must match "
                             "dimension of the tree.")

        neighbors = np.zeros((N, k), dtype=int)
        distances = np.empty((N, k), dtype=float)
        distances.fill(np.inf)

        for i in range(N):
            self.head_node.query(X[i], neighbors[i], distances[i])

        return distances, neighbors

class Node:
    def __init__(self, X, indices, leaf_size, dm):
        self.X = X
        self.indices = indices
        self.leaf_size = leaf_size
        self.dm = dm
        self._divide()

    def _divide(self):
        Xind = self.X[self.indices]
        self.centroid = Xind.mean(0)
        self.radius = self.dm.cdist([self.centroid], Xind)[0].max()
        
        if len(self.indices) > self.leaf_size:
            self.is_leaf = False

            # find the dimension with maximum spread
            maxes = Xind.max(0)
            mins = Xind.min(0)
            i_split = np.argmax(maxes - mins)

            # sort indices on the split dimension
            i_sort = np.argsort(Xind[:, i_split])
            self.indices[:] = self.indices[i_sort]

            # split nodes about the middle
            Nsplit = len(self.indices) / 2
            self.children = (Node(self.X, self.indices[:Nsplit],
                                  self.leaf_size, self.dm),
                             Node(self.X, self.indices[Nsplit:],
                                  self.leaf_size, self.dm))
        else:
            self.is_leaf = True
            self.children = tuple()


    def query(self, pt, neighbors, distances):
        """
        Query the node
        """
        k = len(neighbors)
        
        d = self.dm.dist(pt, self.centroid)

        min_dist = max(0, d - self.radius)
        max_dist = d + self.radius

        if min_dist > distances[-1]:
            # node is too far away: trim it
            return
        elif self.is_leaf:
            # leaf node: compute distances and add to queue
            Xind = self.X[self.indices]
            dist = self.dm.cdist([pt], Xind)[0]

            n = np.concatenate([neighbors, self.indices])
            d = np.concatenate([distances, dist])

            i_sort = np.argsort(d)

            distances[:] = d[i_sort[:k]]
            neighbors[:] = n[i_sort[:k]]
        else:
            self.children[0].query(pt, neighbors, distances)
            self.children[1].query(pt, neighbors, distances)


if __name__ == '__main__':
    from time import time
    
    rseed = np.random.randint(100000)
    print "rseed = %i" % rseed
    np.random.seed(rseed)
    X = np.random.random((1000, 3))
    y = np.random.random((10, 3))

    t0 = time()
    SBT = SlowBallTree(X, leaf_size=2)
    d1, n1 = SBT.query(y, 3)
    t1 = time()

    print "python: %.2g sec" % (t1 - t0)

    t0 = time()
    BT = BallTree(X, leaf_size=2)
    d2, n2 = BT.query(y, 3)
    t1 = time()

    print "cython: %.2g sec" % (t1 - t0)

    print "neighbors match:", np.allclose(n1, n2)
    print "distances match:", np.allclose(d1, d2)
