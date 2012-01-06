This is the outline of a framework that will allow general distance
metrics to be incorporated into scikit-learn BallTree.  The idea is
that we need a fast way to compute the distance between two points
under a given metric.  In the basic framework here, this involves creating
an object which exposes C-pointers to a function and a parameter structure
so that the distance function can be called from either python or directly
from cython with no python overhead.

Framework
---------
I am trying to mimic as much as possible the metrics available in
`scipy.spatial.distance`, matching the speed of these computations,
while maintaining a c function which can be exposed for fast distance
computation within cython loops.  Once this is complete, we will be
able to re-write BallTree to use any of these distance metrics,
and then possibly modify `scipy.spatial.cKDtree` as well.

Benchmarks
----------
run bench.py for a comparison of runtimes between pyDistance and scipy.spatial
for the available metrics.  So far, we're doing pretty well: very close in
most cases.  One exception is boolean values: I need to figure out how to do
fast boolean computations within cython.