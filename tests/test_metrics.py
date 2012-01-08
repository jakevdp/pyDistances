import os, sys
sys.path.append(os.path.abspath('../'))

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from distmetrics import DistanceMetric
from brute_neighbors import brute_force_neighbors
from sklearn.neighbors import NearestNeighbors
import itertools

import cPickle

# dimension for the tests
DTEST = 10

# create some inverse covariance matrices for mahalanobis
VI = np.random.random((DTEST, DTEST))
VI1 = np.dot(VI, VI.T)
VI2 = np.dot(VI.T, VI)

# For each distance metric, specify a dictionary of ancillary parameters
#  and a sequence of values to test.
#
# note that wminkowski and canberra fail in scipy.spacial.cdist/pdist
#  in scipy <= 0.8

# create a user-defined metric
def user_metric(x, y):
    return np.dot(x[::-1], y)

METRIC_DICT = {'euclidean': {},
               'minkowski': dict(p=(1, 1.5, 2.0, 3.0)),
               #'wminkowski': dict(p=(1, 1.5, 2.0),
               #                   w=(np.random.random(DTEST),)),
               'mahalanobis': dict(VI = (None, VI1, VI2)),
               'seuclidean': dict(V = (None, np.random.random(DTEST),)),
               'sqeuclidean': {},
               'cosine': {},
               'correlation': {},
               'hamming': {},
               'jaccard': {},
               'chebyshev': {},
               #'canberra': {},
               'braycurtis': {},
               user_metric: {}}

BOOL_METRIC_DICT = {'yule' : {},
                    'matching' : {},
                    'dice': {},
                    'kulsinski': {},
                    'rogerstanimoto': {},
                    'russellrao': {},
                    'sokalmichener': {},
                    'sokalsneath': {}}


def test_cdist(m1=15, m2=20, rseed=0):
    """Compare DistanceMetric.cdist to scipy.spatial.distance.cdist"""
    np.random.seed(rseed)
    X1 = np.random.random((m1, DTEST))
    X2 = np.random.random((m2, DTEST))
    for (metric, argdict) in METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            dist_metric = DistanceMetric(metric, **kwargs)

            Y1 = dist_metric.cdist(X1, X2)
            Y2 = cdist(X1, X2, metric, **kwargs)

            if not np.allclose(Y1, Y2):
                print metric, keys, vals
                print Y1[:5, :5]
                print Y2[:5, :5]
                assert np.allclose(Y1, Y2)


def test_pdist(m=15, rseed=0):
    """Compare DistanceMetric.pdist to scipy.spatial.distance.pdist"""
    np.random.seed(rseed)
    X = np.random.random((m, DTEST))
    for (metric, argdict) in METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            dist_metric = DistanceMetric(metric, **kwargs)

            Y1 = dist_metric.pdist(X)
            Y2 = pdist(X, metric, **kwargs)

            if not np.allclose(Y1, Y2):
                print metric, keys, vals
                print Y1[:5, :5]
                print Y2[:5, :5]
                assert np.allclose(Y1, Y2)


def test_cdist_bool(m1=15, m2=20, rseed=0):
    """Compare DistanceMetric.cdist to scipy.spatial.distance.cdist"""
    np.random.seed(rseed)
    X1 = (np.random.random((m1, DTEST)) > 0.5).astype(float)
    X2 = (np.random.random((m2, DTEST)) > 0.5).astype(float)
    for (metric, argdict) in BOOL_METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            dist_metric = DistanceMetric(metric, **kwargs)

            Y1 = dist_metric.cdist(X1, X2)
            Y2 = cdist(X1, X2, metric, **kwargs)

            if not np.allclose(Y1, Y2):
                print metric, keys, vals
                print Y1[:5, :5]
                print Y2[:5, :5]
                assert np.allclose(Y1, Y2)


def test_pdist_bool(m=15, rseed=0):
    """Compare DistanceMetric.pdist to scipy.spatial.distance.pdist"""
    np.random.seed(rseed)
    X = (np.random.random((m, DTEST)) > 0.5).astype(float)
    for (metric, argdict) in BOOL_METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            dist_metric = DistanceMetric(metric, **kwargs)

            Y1 = dist_metric.pdist(X)
            Y2 = pdist(X, metric, **kwargs)

            if not np.allclose(Y1, Y2):
                print metric, keys, vals
                print Y1[:5, :5]
                print Y2[:5, :5]
                assert np.allclose(Y1, Y2)

def test_pdist_squareform(m=10, d=3, rseed=0):
    X = np.random.random((m, d))
    dist_metric = DistanceMetric()
    Y1 = squareform(dist_metric.pdist(X, squareform=False))
    Y2 = dist_metric.pdist(X, squareform=True)
    assert np.allclose(Y1, Y2)

def test_user_metric(m1 = 2, m2 = 3):
    X1 = np.random.random((m1, DTEST))
    X2 = np.random.random((m2, DTEST))
    f = lambda x, y: np.dot(x[::-1], y)

    dist_metric = DistanceMetric(f)
    res1 = dist_metric.cdist(X1, X2)

    res2 = cdist(X1, X2, f)

    assert np.allclose(res1, res2)


def test_brute_neighbors(n1=10, n2=20, m=5, k=5, rseed=0):
    X = np.random.random((n1, m))
    Y = np.random.random((n2, m))

    nbrs = NearestNeighbors(k).fit(Y)
    ind1 = nbrs.kneighbors(X, return_distance=False)

    ind2 = brute_force_neighbors(X, Y, k)

    assert np.all(ind1 == ind2)

def test_pickle(n=5, d=3):
    X = np.random.random((n, d))
    VI = np.random.random((d, d))
    VI = np.dot(VI, VI.T)
    
    dm = DistanceMetric('mahalanobis', VI=VI)
    dist = dm.pdist(X)
    
    for protocol in (0, 1, 2):
        s = cPickle.dumps(dm)
        dm2 = cPickle.loads(s)

        assert np.allclose(dist, dm2.pdist(X))


if __name__ == '__main__':
    import nose
    nose.runmodule()
