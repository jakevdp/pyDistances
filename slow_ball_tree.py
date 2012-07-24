"""
Pure Python BallTree
--------------------

This is a pure python ball tree.  It is slow, but should give correct
results, and is much more readable than the cython version of the
ball tree code.

For consistency, it uses the same DistanceMetric object as the cython
BallTree in order to enable the use of arbitrary distance metrics.
"""
import numpy as np
from distmetrics import DistanceMetric

class SlowBallTree:
    def __init__(self, X, leaf_size=20, metric='euclidean', **kwargs):
        self.X = np.asarray(X)
        self.leaf_size = leaf_size
        self.metric = metric
        self.kwargs = kwargs
        
        # create the distance metric
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

        for i in range(N):
            distances[i], neighbors[i] = self.head_node.query(X[i], k, [], [])

        return distances, neighbors

    def query_dual(self, X, k):
        bt = X
        if not isinstance(X, SlowBallTree):
            if X is self.X:
                bt = self
            else:
                bt = SlowBallTree(X, leaf_size=self.leaf_size,
                                  metric=self.metric, **self.kwargs)

        neighbors = np.zeros((X.shape[0], k), dtype=int)
        distances = np.zeros((X.shape[0], k), dtype=float)
        distances.fill(np.inf)

        self.head_node.query_dual(bt.head_node, distances, neighbors)

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


    def query(self, pt, k, distances, neighbors):
        """
        Query the node.
        pt is the query point, neighbors and distances are the current
        set of closest neighbors and distances for the point
        """
        d = self.dm.dist(pt, self.centroid)

        min_dist = max(0, d - self.radius)
        #max_dist = d + self.radius

        if (len(distances) == k) and (min_dist > distances[-1]):
            # node is too far away: trim it
            pass
        elif self.is_leaf:
            # leaf node: compute distances and add to queue
            Xind = self.X[self.indices]
            dist = self.dm.cdist([pt], Xind)[0]

            n = np.concatenate([neighbors, self.indices])
            d = np.concatenate([distances, dist])

            i_sort = np.argsort(d)

            distances = d[i_sort[:k]]
            neighbors = n[i_sort[:k]]
        else:
            distances, neighbors = self.children[0].query(pt, k,
                                                          distances, neighbors)
            distances, neighbors = self.children[1].query(pt, k,
                                                          distances, neighbors)

        return distances, neighbors

    def query_dual(self, node, distances, neighbors):
        """
        Dual-tree query for k nearest neighbors

        node is the other node to query, and [distances, neighbors] are
        of shape (node.X.shape[0], k)
        """
        k = distances.shape[1]
        max_observed = np.max(distances[node.indices, -1])

        d = self.dm.dist(node.centroid, self.centroid)

        min_dist = max(0, d - node.radius - self.radius)
        #max_dist = d + node.radius + self.radius

        if max_observed < min_dist:
            # trim both nodes
            pass

        elif self.is_leaf and node.is_leaf:
            X1ind = node.X[node.indices]
            X2ind = self.X[self.indices]

            dist = self.dm.cdist(X1ind, X2ind)
            nbrs = np.zeros((dist.shape[0], 1), dtype=int) + self.indices

            distances_node = np.hstack([distances[node.indices], dist])
            neighbors_node = np.hstack([neighbors[node.indices], nbrs])

            i_sort = np.argsort(distances_node, 1)[:, :k]
            rng = np.arange(distances_node.shape[0])[:, np.newaxis]

            distances[node.indices] = distances_node[rng, i_sort]
            neighbors[node.indices] = neighbors_node[rng, i_sort]

        elif self.is_leaf:
            self.query_dual(node.children[0], distances, neighbors)
            self.query_dual(node.children[1], distances, neighbors)
        elif node.is_leaf:
            self.children[0].query_dual(node, distances, neighbors)
            self.children[1].query_dual(node, distances, neighbors)
        else:
            self.children[0].query_dual(node.children[0], distances, neighbors)
            self.children[1].query_dual(node.children[0], distances, neighbors)
            self.children[0].query_dual(node.children[1], distances, neighbors)
            self.children[1].query_dual(node.children[1], distances, neighbors)


if __name__ == '__main__':
    from ball_tree import BallTree
    from time import time
    
    rseed = np.random.randint(100000)
    print "rseed = %i" % rseed
    np.random.seed(rseed)
    X = np.random.random((200, 3))
    Y = np.random.random((100, 3))

    t0 = time()
    SBT = SlowBallTree(X, leaf_size=10)
    d1, n1 = SBT.query(Y, 3)
    t1 = time()

    print "python: %.2g sec" % (t1 - t0)

    t0 = time()
    SBT = SlowBallTree(X, leaf_size=10)
    d1a, n1a = SBT.query_dual(Y, 3)
    t1 = time()

    print "python dual: %.2g sec" % (t1 - t0)

    t0 = time()
    BT = BallTree(X, leaf_size=10)
    d2, n2 = BT.query(Y, 3)
    t1 = time()

    print "cython: %.2g sec" % (t1 - t0)

    print "neighbors match:", np.allclose(n1, n2), np.allclose(n1a, n1)
    print "distances match:", np.allclose(d1, d2), np.allclose(d1a, d1)
