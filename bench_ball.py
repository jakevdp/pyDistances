from time import time
import numpy as np
from ball_tree import BallTree

X = np.random.random((10000, 3))

t0 = time()
BT = BallTree(X, 20)
t1 = time()
print "construction: %.2g sec" % (t1 - t0)

for k in 1, 2, 4, 8:
    t0 = time()
    BT.query(X, k)
    t1 = time()
    print "query %i in [%i, %i]: %.3g sec" % (k, X.shape[0], X.shape[1],
                                              t1 - t0)
    
