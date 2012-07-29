from time import time
import numpy as np
from ball_tree import BallTree, KDTree
from sklearn import neighbors

X = np.random.random((20000, 3))
X_query = np.random.random((20000, 3))

t0 = time()
BT = BallTree(X, 30)
t1 = time()
print "BT construction: %.2g sec" % (t1 - t0)

t0 = time()
KDT = KDTree(X, 30)
t1 = time()
print "KDT construction: %.2g sec" % (t1 - t0)

for k in 1, 2, 4, 8:
    print "\nquery %i in [%i, %i]:" % (k, X.shape[0], X.shape[1])
    print "      single     dual"
    t0 = time()
    d1, i1 = BT.query(X_query, k, dualtree=False)
    t1 = time()
    d1, i1 = BT.query(X_query, k, dualtree=True)
    t2 = time()
    print "  BT: %.3g sec   %.3g sec" % (t1 - t0, t2 - t1)

    d2, i2 = KDT.query(X_query, k, dualtree=False)
    t3 = time()
    d2, i2 = KDT.query(X_query, k, dualtree=True)
    t4 = time()
    print "  KDT: %.3g sec   %.3g sec" % (t3 - t2, t4 - t3)
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
