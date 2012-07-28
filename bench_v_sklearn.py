from time import time
import numpy as np
from ball_tree import BallTree as pyBallTree
from sklearn.neighbors import BallTree as skBallTree

X = np.random.random((10000, 3))

t0 = time()
pyBT = pyBallTree(X, 30)
t1 = time()
print "py construction: %.2g sec" % (t1 - t0)

t0 = time()
skBT = skBallTree(X, 30)
t1 = time()
print "sk construction: %.2g sec" % (t1 - t0)

for k in [1, 2, 4, 8]:
    print "query %i in [%i, %i]:" % (k, X.shape[0], X.shape[1])

    t0 = time()
    pyBT.query(X, k, dualtree=False)
    t1 = time()
    print "   py: %.2g sec" % (t1 - t0)

    t0 = time()
    skBT.query(X, k)
    t1 = time()
    print "   sk: %.2g sec" % (t1 - t0)
    
for r in 0.1, 0.3, 0.5:
    print "query r<%.1f in [%i, %i]:" % (r, X.shape[0], X.shape[1])

    t0 = time()
    pyBT.query_radius(X[:1000], r)
    t1 = time()
    print "   py: %.2g sec" % (t1 - t0)

    t0 = time()
    skBT.query_radius(X[:1000], r)
    t1 = time()
    print "   sk: %.2g sec" % (t1 - t0)
