from time import time
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from distmetrics import distance, DistanceMetric
import itertools


# dimension for the tests
DTEST = 10

# create some inverse covariance matrices for mahalanobis
VI = np.random.random((DTEST, DTEST))
VI1 = np.dot(VI, VI.T)
VI2 = np.dot(VI.T, VI)

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
               'braycurtis': {}}

BOOL_METRIC_DICT = {'yule' : {},
                    'matching' : {},
                    'dice': {},
                    'kulsinski': {},
                    'rogerstanimoto': {},
                    'russellrao': {},
                    'sokalmichener': {},
                    'sokalsneath': {}}

def bench_cdist(m1=100, m2=100, rseed=0):
    np.random.seed(rseed)
    X1 = np.random.random((m1, DTEST))
    X2 = np.random.random((m2, DTEST))
    for (metric, argdict) in METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))

            t0 = time()
            dist_metric = DistanceMetric(metric, **kwargs)
            Y1 = dist_metric.cdist(X1, X2)
            t1 = time()
            Y2 = cdist(X1, X2, metric, **kwargs)
            t2 = time()

            print metric, keys, vals
            if not np.allclose(Y1, Y2):
                print " >>>>>> FAIL: results don't match"
            print " - pyDistances: %.2g sec" % (t1 - t0)
            print " - scipy: %.2g sec" % (t2 - t1)


def bench_cdist_bool(m1=100, m2=100, rseed=0):
    np.random.seed(rseed)
    X1 = (np.random.random((m1, DTEST)) > 0.5).astype(float)
    X2 = (np.random.random((m2, DTEST)) > 0.5).astype(float)
    for (metric, argdict) in BOOL_METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))

            t0 = time()
            dist_metric = DistanceMetric(metric, **kwargs)
            Y1 = dist_metric.cdist(X1, X2)
            t1 = time()
            Y2 = cdist(X1, X2, metric, **kwargs)
            t2 = time()

            print metric, keys, vals
            if not np.allclose(Y1, Y2):
                print " >>>>>> FAIL: results don't match"
            print " - pyDistances: %.2g sec" % (t1 - t0)
            print " - scipy: %.2g sec" % (t2 - t1)


if __name__ == '__main__':
    bench_cdist()
    bench_cdist_bool()
