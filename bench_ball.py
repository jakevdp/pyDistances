from time import time
import numpy as np
from ball_tree import BallTree

X = np.random.random((10000, 3))

t0 = time()
BT = BallTree(X, 20)
t1 = time()
print "construction: %.2g sec" % (t1 - t0)

for k in 1, 2, 4, 8:
    for dual in (True, False):
        t0 = time()
        BT.query(X, k, dualtree=dual)
        t1 = time()
        if dual:
            dual_str = ' (dual)'
        else:
            dual_str = ''
        print "query %i in [%i, %i]%s: %.3g sec" % (k, X.shape[0],
                                                    X.shape[1],
                                                    dual_str,
                                                    t1 - t0)
    
for r in 0.1, 0.3, 0.5:
    t0 = time()
    BT.query_radius(X[:1000], r)
    t1 = time()
    print "query r<%.1f in [%i, %i]: %.3g sec" % (r, X.shape[0], X.shape[1],
                                                  t1 - t0)
