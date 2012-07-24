import numpy as np
from numpy.testing import assert_array_almost_equal
from ball_tree import BallTree
from distmetrics import DistanceMetric

from scipy.spatial import cKDTree
from sklearn import neighbors
import itertools

import cPickle

# create a user-defined metric
def user_metric(x, y):
    return abs(x - y).sum()

class TestBallTree:
    def __init__(self, n_samples=100, n_features=5, rseed=0):
        np.random.seed(rseed)
        self.X = np.random.random((n_samples, n_features))
        self.Xbool = (np.random.random(size=(10, 10)) >= 0.5).astype(float)
        self.Ybool = (np.random.random(size=(10, 10)) >= 0.5).astype(float)
        
        VI = np.random.random((n_features, n_features))
        VI1 = np.dot(VI, VI.T)
        VI2 = np.dot(VI.T, VI)
        w = np.random.rand(n_features)

        # For each distance metric, specify a dictionary of ancillary parameters
        #  and a sequence of values to test.
        #
        # note that wminkowski and canberra fail in scipy.spacial.cdist/pdist
        #  in scipy <= 0.8
        self.metrics = {'euclidean': {},
                        'cityblock': {},
                        'minkowski': dict(p=(1, 1.5, 2.0, 3.0)),
                        'wminkowski': dict(p=(1, 1.5, 2.0),
                                           w=(np.random.rand(n_features),)),
                        'mahalanobis': dict(VI = (None, VI1, VI2)),
                        'seuclidean': dict(V = (None, w,)),
                        #'cosine': {},
                        #'correlation': {},
                        'hamming': {},
                        'chebyshev': {},
                        'canberra': {},
                        'braycurtis': {},
                        user_metric: {}}

        self.bool_metrics = {'yule' : {},
                             'matching' : {},
                             'hamming' : {},
                             'jaccard' : {},
                             'dice': {},
                             'kulsinski': {},
                             'rogerstanimoto': {},
                             'russellrao': {},
                             'sokalmichener': {},
                             'sokalsneath': {}
                             }

    def _check_query_knn(self, bt, kdt, k):
        dist_bt, ind_bt = bt.query(self.X, k=k)
        dist_kd, ind_kd = kdt.query(self.X, k=k)

        if k == 1:
            dist_kd = dist_kd[:, None]
            ind_kd = ind_kd[:, None]

        assert_array_almost_equal(dist_bt, dist_kd)
        assert_array_almost_equal(ind_bt, ind_kd)

    def test_query_knn(self):
        bt = BallTree(self.X)
        kdt = cKDTree(self.X)
        for k in (1, 2, 4, 8):
            yield (self._check_query_knn, bt, kdt, k)

    def _check_metrics_float(self, k, metric, kwargs):
        bt = BallTree(self.X, metric=metric, **kwargs)
        dist_bt, ind_bt = bt.query(self.X, k=k)

        dm = DistanceMetric(metric=metric, **kwargs)
        D = dm.pdist(self.X, squareform=True)

        ind_dm = np.argsort(D, 1)[:, :k]
        dist_dm = D[np.arange(self.X.shape[0])[:, None], ind_dm]

        # we don't check the indices here because if there is a tie for
        # nearest neighbor, then the test may fail.  Distances will reflect
        # whether the search was successful
        assert_array_almost_equal(dist_bt, dist_dm)
    
    def test_metrics_float(self, k=5):    
        for (metric, argdict) in self.metrics.iteritems():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                yield (self._check_metrics_float, k, metric, kwargs)

    def _check_metrics_bool(self, k, metric, kwargs):
        bt = BallTree(self.Xbool, metric=metric, **kwargs)
        dist_bt, ind_bt = bt.query(self.Ybool, k=k)

        dm = DistanceMetric(metric=metric, **kwargs)
        D = dm.cdist(self.Ybool, self.Xbool)

        ind_dm = np.argsort(D, 1)[:, :k]
        dist_dm = D[np.arange(self.Ybool.shape[0])[:, None], ind_dm]
        
        # we don't check the indices here because there are very often
        # ties for nearest neighbors, which cause the test to fail.
        # Distances will be correct in either case.
        assert_array_almost_equal(dist_bt, dist_dm)

    def test_ball_tree_metrics_bool(self, k=3):    
        for (metric, argdict) in self.bool_metrics.iteritems():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                yield (self._check_metrics_bool, k, metric, kwargs)

    def _check_p_distance_vs_KDT(self, p):
        bt = BallTree(self.X, leaf_size=10, metric='minkowski', p=p)
        kdt = cKDTree(self.X, leafsize=10)

        dist_bt, ind_bt = bt.query(self.X, k=5)
        dist_kd, ind_kd = kdt.query(self.X, k=5, p=p)

        assert_array_almost_equal(dist_bt, dist_kd)

    def test_p_distance_vs_KDT(self):
        for p in (1, 2, 3, 4, np.inf):
            yield (self._check_p_distance_vs_KDT, p)

    def test_query_radius_count(self, n_samples=100, n_features=10):
        X = 2 * np.random.random(size=(n_samples, n_features)) - 1

        dm = DistanceMetric()
        D = dm.pdist(X, squareform=True)

        r = np.mean(D)

        bt = BallTree(X)
        count1 = bt.query_radius(X, r, count_only=True)

        count2 = (D <= r).sum(1)

        assert_array_almost_equal(count1, count2)

    def test_query_radius_indices(self, n_samples=100, n_features=10):
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

    def test_query_radius_distance(self, n_samples=100, n_features=10):
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

    def _check_pickle(self, protocol, bt1, ind1, dist1):
        s = cPickle.dumps(bt1, protocol=protocol)
        bt2 = cPickle.loads(s)
        ind2, dist2 = bt2.query(self.X)
        assert np.all(ind1 == ind2)
        assert_array_almost_equal(dist1, dist2)
        

    def test_pickle(self):
        bt1 = BallTree(self.X, leaf_size=1)
        ind1, dist1 = bt1.query(self.X)
        for protocol in (0, 1, 2):
            yield (self._check_pickle, protocol, bt1, ind1, dist1)

if __name__ == '__main__':
    import nose
    nose.runmodule()
