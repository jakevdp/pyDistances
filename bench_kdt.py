from time import time
import numpy as np
from ball_tree import BallTree, KDTree

X = np.random.random((10000, 3))

t0 = time()
BT = BallTree(X, 30)
t1 = time()
print "BT construction: %.2g sec" % (t1 - t0)

t0 = time()
KDT = KDTree(X, 30)
t1 = time()
print "KDT construction: %.2g sec" % (t1 - t0)

for k in 1, 2, 4, 8:
    t0 = time()
    d1, i1 = BT.query(X, k, dualtree=True)
    t1 = time()
    d2, i2 = KDT.query(X, k, dualtree=True)
    t2 = time()
    print "query %i in [%i, %i]:" % (k, X.shape[0], X.shape[1])
    print "  BT: %.3g sec" % (t1 - t0)
    print "  KDT: %.3g sec" % (t2 - t1)
    print "       (results match: %s)" % np.allclose(d1, d2)

#for r in 0.1, 0.3, 0.5:
#    for tree in (BT, KDT):
#        t0 = time()
#        BT.query_radius(X[:1000], r)
#        t1 = time()
#        print tree.__class__.__name__
#        print "  query r<%.1f in [%i, %i]: %.3g sec" % (r, X.shape[0],
#                                                        X.shape[1],
#                                                        t1 - t0)
