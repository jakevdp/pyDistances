cimport numpy as np
import numpy as np

from distmetrics cimport DistanceMetric
from distmetrics import DTYPE

def brute_force_neighbors(X, Y, k, metric='euclidean', **kwargs):
    """
    use brute force neighbors to find the neighbors of X in Y
    """
    X = np.asarray(X, dtype=DTYPE)
    Y = np.asarray(Y, dtype=DTYPE)

    distances = DistanceMetric(metric, **kwargs).cdist(X, Y)

    indices = distances.argsort(1)

    return indices[:, :k]
