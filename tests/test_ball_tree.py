import os, sys
sys.path.append(os.path.abspath('../'))

import numpy as np
from numpy.testing import assert_array_almost_equal
from ball_tree import BallTree
from distmetrics import DistanceMetric

from scipy.spatial import cKDTree
from sklearn import neighbors
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
    return abs(x - y).sum()

METRIC_DICT = {'euclidean': {},
               'cityblock': {},
               'minkowski': dict(p=(1, 1.5, 2.0, 3.0)),
               'wminkowski': dict(p=(1, 1.5, 2.0),
                                  w=(np.random.random(DTEST),)),
               'mahalanobis': dict(VI = (None, VI1, VI2)),
               'seuclidean': dict(V = (None, np.random.random(DTEST),)),
               'cosine': {},
               #'correlation': {},
               #'hamming': {},
               'chebyshev': {},
               'canberra': {},
               'braycurtis': {},
               user_metric: {}}

BOOL_METRIC_DICT = {'yule' : {},
                    'matching' : {},
                    'hamming' : {},
                    'jaccard' : {},
                    'dice': {},
                    'kulsinski': {},
                    'rogerstanimoto': {},
                    'russellrao': {},
                    'sokalmichener': {},
                    'sokalsneath': {}}


def test_ball_tree_query():
    X = np.random.random(size=(100, 5))

    for k in (2, 4, 6):
        bt = BallTree(X)
        kdt = cKDTree(X)

        dist_bt, ind_bt = bt.query(X, k=k)
        dist_kd, ind_kd = kdt.query(X, k=k)

        assert_array_almost_equal(dist_bt, dist_kd)

def test_ball_tree_metrics_float():
    X = np.random.random(size=(100, DTEST))
    k = 5
    
    for (metric, argdict) in METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            bt = BallTree(X, metric=metric, **kwargs)
            dist_bt, ind_bt = bt.query(X, k=k)

            dm = DistanceMetric(metric=metric, **kwargs)
            D = dm.pdist(X, squareform=True)
            ind_dm = np.argsort(D, 1)[:, :k]
            dist_dm = D[np.arange(X.shape[0])[:, None], ind_dm]

            try:
                assert_array_almost_equal(dist_bt, dist_dm)
            except:
                print metric, kwargs
                assert_array_almost_equal(dist_bt, dist_dm)

def test_ball_tree_metrics_bool():
    X = (np.random.random(size=(10, 10)) >= 0.5).astype(float)
    Y = (np.random.random(size=(10, 10)) >= 0.5).astype(float)
    k = 1
    
    for (metric, argdict) in BOOL_METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            bt = BallTree(X, metric=metric, **kwargs)
            dist_bt, ind_bt = bt.query(Y, k=k)

            dm = DistanceMetric(metric=metric, **kwargs)
            D = dm.cdist(Y, X)

            ind_dm = np.argsort(D, 1)[:, :k]
            dist_dm = D[np.arange(Y.shape[0])[:, None], ind_dm]

            try:
                assert_array_almost_equal(dist_bt, dist_dm)
            except:
                print metric, kwargs
                assert_array_almost_equal(dist_bt, dist_dm)


def test_ball_tree_p_distance():
    X = np.random.random(size=(100, 5))

    for p in (1, 2, 3, 4, np.inf):
        bt = BallTree(X, leaf_size=10, metric='minkowski', p=p)
        kdt = cKDTree(X, leafsize=10)

        dist_bt, ind_bt = bt.query(X, k=5)
        dist_kd, ind_kd = kdt.query(X, k=5, p=p)

        assert_array_almost_equal(dist_bt, dist_kd)


def test_ball_tree_query_radius_count(n_samples=100, n_features=10):
    X = 2 * np.random.random(size=(n_samples, n_features)) - 1

    dm = DistanceMetric()
    D = dm.pdist(X, squareform=True)

    r = np.mean(D)

    bt = BallTree(X)
    count1 = bt.query_radius(X, r, count_only=True)

    count2 = (D <= r).sum(1)

    assert_array_almost_equal(count1, count2)


def test_ball_tree_query_radius_indices(n_samples=100, n_features=10):
    X = 2 * np.random.random(size=(n_samples, n_features)) - 1

    dm = DistanceMetric()
    D = dm.cdist(X[:10], X)

    r = np.mean(D)

    bt = BallTree(X)
    ind = bt.query_radius(X[:10], r, return_distance=False)

    for i in range(10):
        ind1 = ind[i]
        ind2 = np.where(D[i] <= r)[0]

        ind1.sort()
        ind2.sort()

        assert_array_almost_equal(ind1, ind2)    


def test_ball_tree_query_radius_distance(n_samples=100, n_features=10):
    X = 2 * np.random.random(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1E-15  # roundoff error can cause test to fail
    bt = BallTree(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind, dist = bt.query_radius(query_pt, r + eps, return_distance=True)

        ind = ind[0]
        dist = dist[0]

        d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))

        assert_array_almost_equal(d, dist)


def test_ball_tree_pickle():
    X = np.random.random(size=(10, 3))
    bt1 = BallTree(X, leaf_size=1)
    ind1, dist1 = bt1.query(X)
    for protocol in (0, 1, 2):
        s = cPickle.dumps(bt1, protocol=protocol)
        bt2 = cPickle.loads(s)
        ind2, dist2 = bt2.query(X)
        assert np.all(ind1 == ind2)
        assert_array_almost_equal(dist1, dist2)


if __name__ == '__main__':
    import nose
    nose.runmodule()
