import numpy as np
from distmetrics import Pairwise
from pairwise import pairwise6

from time import time

pw = Pairwise("euclidean")
#pw = Pairwise("raw")
pairwise = pw.cdist

descr = ['Pairwise() class',
         'memview / no slicing / with function call']
funcs = [pairwise, pairwise6]
results = []


X = np.random.random((2000, 3))

print "Computing %i x %i distances in %i dimensions" % (X.shape[0],
                                                        X.shape[0],
                                                        X.shape[1])

for i in range(len(funcs)):
    func = funcs[i]
    t0 = time()
    d = func(X, X)
    t1 = time()

    results.append(d)
    print i + 1, ':', descr[i]
    print "    time: %.3g sec" % (t1 - t0)

print
print "check that results match:"
for i in range(len(results) - 1):
    print "  ", i+1, 'vs.', i+2, ':', 
    if np.allclose(results[i], results[i + 1]):
        print "match"
    else:
        print "DO NOT MATCH"
