import itertools

import numpy as np
from numpy.testing import assert_array_almost_equal

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from pyDistances.distmetrics import DistanceMetric

def user_metric(x, y):
    return np.sum(abs(x - y) ** 3)

def pdist_conv(d, **kwargs):
    return d ** kwargs['p']

def sqdist_conv(d, **kwargs):
    return d * d

class TestMetrics:
    def __init__(self, n1=5, n2=6, d=4, zero_frac=0.5,
                 rseed=0, dtype=np.float64):
        np.random.seed(rseed)
        self.X1 = np.random.random((n1, d)).astype(dtype)
        self.X2 = np.random.random((n2, d)).astype(dtype)

        self.X1[self.X1 < zero_frac] = 0
        self.X2[self.X2 < zero_frac] = 0

        self.spX1 = csr_matrix(self.X1)
        self.spX2 = csr_matrix(self.X2)

        VI = np.random.random((d, d))
        VI = np.dot(VI, VI.T)

        w = np.random.random(d)

        self.scipy_metrics = {'minkowski':dict(p=(1, 1.5, 2, 3)),
                              'wminkowski':dict(p=(1, 1.5, 2, 3),
                                                w=(w,)),
                              'mahalanobis':dict(VI=(None, VI)),
                              'seuclidean':dict(V=(None, w)),
                              'euclidean':{},
                              'cityblock':{},
                              'sqeuclidean':{},
                              'cosine':{},
                              'correlation':{},
                              'chebyshev':{},
                              'canberra':{},
                              'braycurtis':{},
                              'hamming':{},
                              'jaccard':{},
                              'yule':{},
                              'matching':{},
                              'dice':{},
                              'kulsinski':{},
                              'rogerstanimoto':{},
                              'russellrao':{},
                              'sokalmichener':{},
                              'sokalsneath':{},
                              user_metric:{}}

        self.reduced_metrics = {'pminkowski': ('minkowski', pdist_conv),
                                'pwminkowski': ('wminkowski', pdist_conv),
                                'sqmahalanobis': ('mahalanobis', sqdist_conv),
                                'sqseuclidean': ('seuclidean', sqdist_conv),
                                'sqeuclidean': ('euclidean', sqdist_conv)}

    def test_cdist(self):
        for metric, argdict in self.scipy_metrics.iteritems():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = cdist(self.X1, self.X2, metric, **kwargs)
                dm = DistanceMetric(metric, **kwargs)
                for X1 in self.X1, self.spX1:
                    for X2 in self.X2, self.spX2:
                        yield self.check_cdist, metric, X1, X2, dm, D_true

        for rmetric, (metric, func) in self.reduced_metrics.iteritems():
            argdict = self.scipy_metrics[metric]
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = func(cdist(self.X1, self.X2, metric, **kwargs),
                              **kwargs)
                dm = DistanceMetric(rmetric, **kwargs)
                for X1 in self.X1, self.spX1:
                    for X2 in self.X2, self.spX2:
                        yield self.check_cdist, rmetric, X1, X2, dm, D_true

    def check_cdist(self, metric, X1, X2, dm, D_true):
        D12 = dm.cdist(X1, X2)
        assert_array_almost_equal(D12, D_true, 6,
                                  "Mismatch for metric=%s, X1=%s, X2=%s"
                                  % (metric,
                                     X1.__class__.__name__,
                                     X2.__class__.__name__))

    def test_pdist(self):
        for metric, argdict in self.scipy_metrics.iteritems():
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = pdist(self.X1, metric, **kwargs)
                Dsq_true = squareform(D_true)
                dm = DistanceMetric(metric, **kwargs)
                for X in self.X1, self.spX1:
                    yield self.check_pdist, metric, X, dm, Dsq_true, True

                for X in self.X1, self.spX1:
                    yield self.check_pdist, metric, X, dm, D_true, False

        for rmetric, (metric, func) in self.reduced_metrics.iteritems():
            argdict = self.scipy_metrics[metric]
            keys = argdict.keys()
            for vals in itertools.product(*argdict.values()):
                kwargs = dict(zip(keys, vals))
                D_true = func(pdist(self.X1, metric, **kwargs),
                              **kwargs)
                Dsq_true = squareform(D_true)
                dm = DistanceMetric(rmetric, **kwargs)
                for X in self.X1, self.spX1:
                    yield self.check_pdist, rmetric, X, dm, Dsq_true, True

                for X in self.X1, self.spX1:
                    yield self.check_pdist, rmetric, X, dm, D_true, False

    def check_pdist(self, metric, X, dm, D_true, squareform):
        D12 = dm.pdist(X, squareform=squareform)

        # set diagonal to zero for non-metrics
        if squareform: D12.flat[::X.shape[0] + 1] = 0
        assert_array_almost_equal(D12, D_true, 6,
                                  "Mismatch for pdist square=%s, "
                                  "metric=%s, X=%s"
                                  % (squareform, metric, X.__class__.__name__))



if __name__ == '__main__':
    import nose
    nose.runmodule()
