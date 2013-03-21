**Note: this module should be considered unsupported.  It probably won't compile,
and if it does, it may contain errors in some cases.  See a more developed version
of this functionality in scikit-learn:** https://github.com/scikit-learn/scikit-learn/pull/1732/

===========
pyDistances
===========
Fast cython distance computations in a variety of metrics which expose
pointers to C functions.

This is a framework that will allow general distance
metrics to be incorporated into tree-based neighbors searches.
The idea is that we need a fast way to compute the distance between two points
under a given metric.  In the basic framework here, this involves creating
an object which exposes C-pointers to a function and a parameter structure
so that the distance function can be called from python as normal, 
or alternatively can be called directly from cython without python overhead.

``pdist``/``cdist``
-------------------
The code has functions which duplicate the behavior of
``scipy.spatial.distances.pdist`` and ``scipy.spatial.distances.cdist``.

BallTree
--------
The code features a BallTree object which can quickly return nearest neighbors
under any of the available metrics.

Benchmarks
----------
run bench.py for a comparison of runtimes between pyDistance and scipy.spatial
for the available metrics.  Times are comparable in most cases, much faster
in a few cases, and slightly slower in a few cases.

TODO
----
Search TODO within distmetrics.pyx to see a list.  One big one (which should
be straightforward) is to make the distance metrics work with CSR matrices.
This will involve writing an alternate version of each core distance function.
