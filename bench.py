import itertools
from time import time
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from pyDistances.distmetrics import DistanceMetric



# dimension for the tests
DTEST = 10

# create some inverse covariance matrices for mahalanobis
VI = np.random.random((DTEST, DTEST))
VI1 = np.dot(VI, VI.T)
VI2 = np.dot(VI.T, VI)

# create a user-defined metric
def user_metric(x, y):
    return np.dot(x[::-1], y)

METRIC_DICT = {'euclidean': {},
               'cityblock': {},
               'minkowski': dict(p=(1, 1.5, 2.0, 3.0)),
               'wminkowski': dict(p=(1, 1.5, 2.0),
                                  w=(np.random.random(DTEST),)),
               'mahalanobis': dict(VI = (None, VI1, VI2)),
               'seuclidean': dict(V = (None, np.random.random(DTEST),)),
               'sqeuclidean': {},
               'cosine': {},
               'correlation': {},
               'chebyshev': {},
               'canberra': {},
               'braycurtis': {},
               user_metric: {}}

BOOL_METRIC_DICT = {'hamming': {},
                    'jaccard': {},
                    'yule' : {},
                    'matching' : {},
                    'dice': {},
                    'kulsinski': {},
                    'rogerstanimoto': {},
                    'russellrao': {},
                    'sokalmichener': {},
                    'sokalsneath': {}}

def param_info(kwargs):
    s = '('
    for key,val in kwargs.iteritems():
        s += key
        s += '='
        if type(val) == np.ndarray:
            s += '[%s array], ' % 'x'.join(map(str,val.shape))
        else:
            s += '%s, ' % str(val)
    s += ')'
    return s

def bench_float(m1=200, m2=200, rseed=0):
    print 79 * '_'
    print " real valued distance metrics"
    print
    np.random.seed(rseed)
    X1 = np.random.random((m1, DTEST))
    X2 = np.random.random((m2, DTEST))
    for (metric, argdict) in METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            print metric, param_info(kwargs)

            t0 = time()
            try:
                dist_metric = DistanceMetric(metric, **kwargs)
                Yc1 = dist_metric.cdist(X1, X2)
            except Exception as inst:
                print " >>>>>>>>>> error in pyDistances cdist:"
                print "           ", inst
            t1 = time()
            try:
                Yc2 = cdist(X1, X2, metric, **kwargs)
            except Exception as inst:
                print " >>>>>>>>>> error in scipy cdist:"
                print "           ", inst
            t2 = time()
            try:
                dist_metric = DistanceMetric(metric, **kwargs)
                Yp1 = dist_metric.pdist(X1)
            except Exception as inst:
                print " >>>>>>>>>> error in pyDistances pdist:"
                print "           ", inst
            t3 = time()
            try:
                Yp2 = pdist(X1, metric, **kwargs)
            except Exception as inst:
                print " >>>>>>>>>> error in scipy pdist:"
                print "           ", inst
            t4 = time()

            if not np.allclose(Yc1, Yc2):
                print " >>>>>>>>>> FAIL: cdist results don't match"
            if not np.allclose(Yp1, Yp2):
                print " >>>>>>>>>> FAIL: pdist results don't match"
            print " - pyDistances:  c: %.4f sec     p: %.4f sec" % (t1 - t0,
                                                                    t3 - t2)
            print " - scipy:        c: %.4f sec     p: %.4f sec" % (t2 - t1,
                                                                    t4 - t3)

    print ''


def bench_bool(m1=200, m2=200, rseed=0):
    print 79 * '_'
    print " boolean distance metrics"
    print
    np.random.seed(rseed)
    X1 = (np.random.random((m1, DTEST)) > 0.5).astype(float)
    X2 = (np.random.random((m2, DTEST)) > 0.5).astype(float)
    for (metric, argdict) in BOOL_METRIC_DICT.iteritems():
        keys = argdict.keys()
        for vals in itertools.product(*argdict.values()):
            kwargs = dict(zip(keys, vals))
            print metric, param_info(kwargs)

            t0 = time()
            try:
                dist_metric = DistanceMetric(metric, **kwargs)
                Yc1 = dist_metric.cdist(X1, X2)
            except Exception as inst:
                print " >>>>>>>>>> error in pyDistances cdist:"
                print "           ", inst
            t1 = time()
            try:
                Yc2 = cdist(X1, X2, metric, **kwargs)
            except Exception as inst:
                print " >>>>>>>>>> error in scipy cdist:"
                print "           ", inst
            t2 = time()
            try:
                dist_metric = DistanceMetric(metric, **kwargs)
                Yp1 = dist_metric.pdist(X1)
            except Exception as inst:
                print " >>>>>>>>>> error in pyDistances pdist:"
                print "           ", inst
            t3 = time()
            try:
                Yp2 = pdist(X1, metric, **kwargs)
            except Exception as inst:
                print " >>>>>>>>>> error in scipy pdist:"
                print "           ", inst
            t4 = time()

            if not np.allclose(Yc1, Yc2):
                print " >>>>>>>>>> FAIL: cdist results don't match"
            if not np.allclose(Yp1, Yp2):
                print " >>>>>>>>>> FAIL: pdist results don't match"
            print " - pyDistances:  c: %.4f sec     p: %.4f sec" % (t1 - t0,
                                                                    t3 - t2)
            print " - scipy:        c: %.4f sec     p: %.4f sec" % (t2 - t1,
                                                                    t4 - t3)

    print ''


if __name__ == '__main__':
    bench_float()
    bench_bool()
