
# Author: Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD

# TODO:
#  - documentation update with metrics
#
#  - code cleanup, make sure all variables are declared
#
#  - currently all metrics are used without precomputed values.  This should
#    be addressed.
#
#  - abstract-out node objects to make the code more readable.  I think it
#    can be done, but not sure if it will lead to a speed penalty...  Create
#    kd-tree node type as well?
#
#  - Add double-tree implementation to speed multiple neighbor searches.
#
#  - query_self method (common case)
#
#  - correlation function query
#
# Thoughts:
#  what about using fibonacci heaps to keep track of visited nodes?
#  would recursion rather than an explicit stack be more readable?

"""
=========
Ball Tree
=========
A ball tree is a data object which speeds up nearest neighbor
searches in high dimensions (see scikit-learn neighbors module
documentation for an overview of neighbor trees). There are many
types of ball trees.  This package provides a basic implementation
in cython.

Implementation Notes
--------------------

A ball tree can be thought of as a collection of nodes.  Each node
stores a centroid, a radius, and the pointers to two child nodes.

* centroid : the centroid of a node is the mean of all the locations
    of points within the node
* radius : the radius of a node is the distance from the centroid
    to the furthest point in the node.
* subnodes : each node has a maximum of 2 child nodes.  The data within
    the parent node is divided between the two child nodes.

In a typical tree implementation, nodes may be classes or structures which
are dynamically allocated as needed.  This offers flexibility in the number
of nodes, and leads to very straightforward and readable code.  It also means
that the tree can be dynamically augmented or pruned with new data, in an
in-line fashion.  This approach generally leads to recursive code: upon
construction, the head node constructs its child nodes, the child nodes
construct their child nodes, and so-on.

For an illustration of this sort of approach, refer to slow_ball_tree.py, which
is a python-only implementation designed for readibility rather than speed.

The current package uses a different approach: all node data is stored in
a set of numpy arrays which are pre-allocated.  The main advantage of this
approach is that the whole object can be quickly and easily saved to disk
and reconstructed from disk.  This also allows for an iterative interface
which gives more control over the heap, and leads to speed.  There are a
few disadvantages, however: once the tree is built, augmenting or pruning it
is not as straightforward.  Also, the size of the tree must be known from the
start, so there is not as much flexibility in building it.

BallTree Storage
~~~~~~~~~~~~~~~~
The BallTree information is stored using a combination of
"Array of Structures" and "Structure of Arrays" to maximize speed.
Given input data of size ``(n_samples, n_features)``, BallTree computes the
expected number of nodes ``n_nodes`` (see below), and allocates the
following arrays:

* ``data`` : a float array of shape ``(n_samples, n_features)``
    This is simply the input data.  If the input matrix is well-formed
    (contiguous, c-ordered, correct data type) then no copy is needed
* ``idx_array`` : an integer array of size ``n_samples``
    This can be thought of as an array of pointers to the data in ``data``.
    Rather than shuffling around the data itself, we shuffle around pointers
    to the rows in data.
* ``node_centroid_arr`` : a float array of shape ``(n_nodes, n_features)``
    This stores the centroid of the data in each node.
* ``node_info_arr`` : a size-``n_nodes`` array of ``NodeInfo`` structures.
    This stores information associated with each node.  Each ``NodeInfo``
    instance has the following attributes:
    - ``idx_start``
    - ``idx_end`` : ``idx_start`` and ``idx_end`` reference the part of
      ``idx_array`` which point to the data associated with the node.
      The data in node with index ``i_node`` is given by
      ``data[idx_array[idx_start:idx_end]]``
    - ``is_leaf`` : a boolean value which tells whether this node is a leaf:
      that is, whether or not it has children.
    - ``radius`` : a floating-point value which gives the distance from
      the node centroid to the furthest point in the node.

One feature here is that there are no stored pointers from parent nodes to
child nodes and vice-versa.  These pointers are implemented implicitly:
For a node with index ``i``, the two children are found at indices
``2 * i + 1`` and ``2 * i + 2``, while the parent is found at index
``floor((i - 1) / 2)``.  The root node has no parent.

With this data structure in place, the functionality of the above BallTree
pseudo-code can be implemented in a much more efficient manner.
Most of the data passing done in this code uses raw data pointers.
Using numpy arrays would be preferable for safety, but the
overhead of array slicing and sub-array construction leads to execution
time which is several orders of magnitude slower than the current
implementation.

Priority Queue vs Max-heap
~~~~~~~~~~~~~~~~~~~~~~~~~~
When querying for more than one neighbor, the code must maintain a list of
the current k nearest points.  The BallTree code implements this in two ways.

- A priority queue: this is just a sorted list.  When an item is added,
  it is inserted in the appropriate location.  The cost of the search plus
  insert averages O[k].
- A max-heap: this is a binary tree structure arranged such that each node is
  greater than its children.  The cost of adding an item is O[log(k)].
  At the end of the iterations, the results must be sorted: a quicksort is
  used, which averages O[k log(k)].  Quicksort has worst-case O[k^2]
  performance, but because the input is already structured in a max-heap,
  the worst case will not be realized.  Thus the sort is a one-time operation
  with cost O[k log(k)].

Each insert is performed an average of log(N) times per query, where N is
the number of training points.  Because of this, for a single query, the
priority-queue approach costs O[k log(N)], and the max-heap approach costs
O[log(k)log(N)] + O[k log(k)].  Tests show that for sufficiently large k,
the max-heap approach out-performs the priority queue approach by a factor
of a few.  In light of these tests, the code uses a priority queue for
k < 5, and a max-heap otherwise.

Memory Allocation
~~~~~~~~~~~~~~~~~
It is desirable to construct a tree in as balanced a way as possible.
Given a training set with n_samples and a user-supplied leaf_size, if
the points in each node are divided as evenly as possible between the
two children, the maximum depth needed so that leaf nodes satisfy
``leaf_size <= n_points <= 2 * leaf_size`` is given by
``n_levels = 1 + max(0, floor(log2((n_samples - 1) / leaf_size)))``
(with the exception of the special case where ``n_samples <= leaf_size``)
For a given number of levels, the number of points in a tree is given by
``n_nodes = 2 ** n_levels - 1``.  Both of these results can be shown
by induction.  Using them, the correct amount of memory can be pre-allocated
for a given ``n_samples`` and ``leaf_size``.
"""

import numpy as np

cimport numpy as np
cimport cython
from libc cimport stdlib
from libc.math cimport fmax, fmin

from distmetrics cimport DistanceMetric, DTYPE_t
from distmetrics import DTYPE

from sklearn.utils import array2d

######################################################################
# global definitions

# type used for indices & counts
# warning: there will be problems if this is switched to an unsigned type!
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

######################################################################
# NodeInfo struct
#  used to keep track of node information.
#  there is also a centroid for each node: this is kept in a separate
#  array for efficiency.  This is a hybrid of the "Array of Structures"
#  and "Structure of Arrays" styles.
cdef struct NodeInfo:
    ITYPE_t idx_start
    ITYPE_t idx_end
    ITYPE_t is_leaf
    DTYPE_t radius


######################################################################
# stack struct
#  This is used to keep track of the recursion stack in Node_query
cdef struct stack_item:
    ITYPE_t i_node
    DTYPE_t reduced_dist_LB


cdef struct stack:
    int n
    stack_item* heap
    int size


@cython.profile(False)
cdef inline void stack_create(stack* self, int size):
    self.size = size
    self.heap = <stack_item*> stdlib.malloc(sizeof(stack_item) * size)
    self.n = 0


@cython.profile(False)
cdef inline void stack_destroy(stack* self):
    stdlib.free(self.heap)


@cython.profile(False)
cdef inline void stack_resize(stack* self, int new_size):
    #print "resize", self.n, new_size
    if new_size < self.n:
        raise ValueError("new_size smaller than current")

    self.size = new_size
    self.heap = <stack_item*>stdlib.realloc(<void*> self.heap,
                                            new_size * sizeof(stack_item))


@cython.profile(False)
cdef inline void stack_push(stack* self, stack_item item):
    if self.n >= self.size:
        stack_resize(self, 2 * self.size + 1)

    self.heap[self.n] = item
    self.n += 1


@cython.profile(False)
cdef inline stack_item stack_pop(stack* self):
    if self.n == 0:
        raise ValueError("popping empty stack")

    self.n -= 1
    return self.heap[self.n]


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)

######################################################################
# Invalid metrics
#
# These are not true metrics (they don't satisfy the triangle inequality)
# so BallTree will not work with them
INVALID_METRICS = ['sqeuclidean', 'correlation', 'pminkowski',
                   'pwminkowski', 'sqseuclidean', 'sqmahalanobis']


######################################################################
# BallTree class
#
cdef class BallTree(object):
    """
    Ball Tree for fast nearest-neighbor searches :

    BallTree(X, leaf_size=20, p=2.0)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        n_samples is the number of points in the data set, and
        n_features is the dimension of the parameter space.
        Note: if X is a C-contiguous array of doubles then data will
        not be copied. Otherwise, an internal copy will be made.

    leaf_size : positive integer (default = 20)
        Number of points at which to switch to brute-force. Changing
        leaf_size will not affect the results of a query, but can
        significantly impact the speed of a query and the memory required
        to store the built ball tree.  The amount of memory needed to
        store the tree scales as
        2 ** (1 + floor(log2((n_samples - 1) / leaf_size))) - 1
        For a specified ``leaf_size``, a leaf node is guaranteed to
        satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in
        the case that ``n_samples < leaf_size``.

    metric : string, function, or float
        distance metric.  See distmetrics docstring for available values.

    p : distance metric for the BallTree.  ``p`` encodes the Minkowski
        p-distance::

            D = sum((X[i] - X[j]) ** p) ** (1. / p)

        p must be greater than or equal to 1, so that the triangle
        inequality will hold.  If ``p == np.inf``, then the distance is
        equivalent to::

            D = max(X[i] - X[j])

    Attributes
    ----------
    data : np.ndarray
        The training data

    Examples
    --------
    Query for k-nearest neighbors

        # >>> import numpy as np
        # >>> np.random.seed(0)
        # >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        # >>> ball_tree = BallTree(X, leaf_size=2)
        # >>> dist, ind = ball_tree.query(X[0], n_neighbors=3)
        # >>> print ind  # indices of 3 closest neighbors
        # [0 3 1]
        # >>> print dist  # distances to 3 closest neighbors
        # [ 0.          0.19662693  0.29473397]

    Pickle and Unpickle a ball tree (using protocol = 2).  Note that the
    state of the tree is saved in the pickle operation: the tree is not
    rebuilt on un-pickling

        # >>> import numpy as np
        # >>> import pickle
        # >>> np.random.seed(0)
        # >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        # >>> ball_tree = BallTree(X, leaf_size=2)
        # >>> s = pickle.dumps(ball_tree, protocol=2)
        # >>> ball_tree_copy = pickle.loads(s)
        # >>> dist, ind = ball_tree_copy.query(X[0], k=3)
        # >>> print ind  # indices of 3 closest neighbors
        # [0 3 1]
        # >>> print dist  # distances to 3 closest neighbors
        # [ 0.          0.19662693  0.29473397]
    """
    cdef readonly np.ndarray data
    cdef np.ndarray idx_array
    cdef np.ndarray node_centroid_arr
    cdef np.ndarray node_info_arr
    cdef ITYPE_t leaf_size
    cdef ITYPE_t n_levels
    cdef ITYPE_t n_nodes

    cdef DistanceMetric dm

    def __cinit__(self):
        """
        initialize all arrays to empty.  This will prevent memory errors
        in rare cases where __init__ is not called
        """
        self.data = np.empty((0,0), dtype=DTYPE)
        self.idx_array = np.empty(0, dtype=ITYPE)
        self.node_centroid_arr = np.empty((0,0), dtype=DTYPE)
        self.node_info_arr = np.empty(0, dtype='c')

    def __init__(self, X, leaf_size=20, 
                 metric="minkowski", p=2, **kwargs):
        if metric in INVALID_METRICS:
            raise ValueError("metric %s does not satisfy the triangle "
                             "inequality: BallTree cannot be used")
        self.data = np.asarray(X, dtype=DTYPE, order='C')

        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if self.data.ndim != 2:
            raise ValueError("X should have two dimensions")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size

        # set up dist_metric if needed
        self.dm = DistanceMetric(metric, p=p, **kwargs)
        if self.dm.learn_params_from_data:
            self.dm.set_params_from_data(self.data)

        cdef ITYPE_t n_samples = self.data.shape[0]
        cdef ITYPE_t n_features = self.data.shape[1]

        # determine number of levels in the ball tree, and from this
        # the number of nodes in the ball tree
        self.n_levels = np.log2(max(1, (n_samples - 1) / self.leaf_size)) + 1
        self.n_nodes = (2 ** self.n_levels) - 1

        self.idx_array = np.arange(n_samples, dtype=ITYPE)

        self.node_centroid_arr = np.empty((self.n_nodes, n_features),
                                          dtype=DTYPE, order='C')

        self.node_info_arr = np.empty(self.n_nodes * sizeof(NodeInfo),
                                      dtype='c', order='C')
        self.build_tree_()

    def __reduce__(self):
        """reduce method used for pickling"""
        return (newObj, (BallTree,), self.__getstate__())

    def __getstate__(self):
        """get state for pickling"""
        return (self.data,
                self.idx_array,
                self.node_centroid_arr,
                self.node_info_arr,
                self.leaf_size,
                self.n_levels,
                self.n_nodes,
                self.dm)

    def __setstate__(self, state):
        """set state for pickling"""
        self.data = state[0]
        self.idx_array = state[1]
        self.node_centroid_arr = state[2]
        self.node_info_arr = state[3]
        self.leaf_size = state[4]
        self.n_levels = state[5]
        self.n_nodes = state[6]
        self.dm = state[7]

    def query(self, X, k=1, return_distance=True):
        """
        query(X, k=1, return_distance=True)

        query the Ball Tree for the k nearest neighbors

        Parameters
        ----------
        X : array-like, last dimension self.n_features
            An array of points to query
        k : integer  (default = 1)
            The number of nearest neighbors to return
        return_distance : boolean (default = True)
            if True, return a tuple (d,i)
            if False, return array i

        Returns
        -------
        i    : if return_distance == False
        (d, i) : if return_distance == True

        d : array of doubles - shape: x.shape[:-1] + (k,)
            each entry gives the sorted list of distances to the
            neighbors of the corresponding point

        i : array of integers - shape: x.shape[:-1] + (k,)
            each entry gives the sorted list of indices of
            neighbors of the corresponding point

        Examples
        --------
        Query for k-nearest neighbors

            # >>> import numpy as np
            # >>> np.random.seed(0)
            # >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
            # >>> ball_tree = BallTree(X, leaf_size=2)
            # >>> dist, ind = ball_tree.query(X[0], k=3)
            # >>> print ind  # indices of 3 closest neighbors
            # [0 3 1]
            # >>> print dist  # distances to 3 closest neighbors
            # [ 0.          0.19662693  0.29473397]
        """
        X = array2d(X, dtype=DTYPE, order='C')

        if X.shape[-1] != self.data.shape[1]:
            raise ValueError("query data dimension must match BallTree "
                             "data dimension")

        if k > self.data.shape[0]:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")

        # flatten X for iteration
        orig_shape = X.shape
        X = X.reshape((-1, X.shape[-1]))

        cdef ITYPE_t i
        cdef ITYPE_t n_neighbors = k
        cdef np.ndarray distances = np.empty((X.shape[0], n_neighbors),
                                             dtype=DTYPE)
        cdef np.ndarray idx_array = np.empty((X.shape[0], n_neighbors),
                                             dtype=ITYPE)
        cdef np.ndarray Xi

        distances.fill(np.inf)

        cdef DTYPE_t* pt
        cdef DTYPE_t* dist_ptr = <DTYPE_t*> distances.data
        cdef ITYPE_t* idx_ptr = <ITYPE_t*> idx_array.data

        cdef DTYPE_t reduced_dist_LB
        cdef NodeInfo* node_info = <NodeInfo*> self.node_info_arr.data
        cdef DTYPE_t* node_centroid = <DTYPE_t*> self.node_centroid_arr.data
        cdef ITYPE_t n_features = self.data.shape[1]
        
        # create heap/queue object for holding results
        cdef HeapBase heap
        if n_neighbors == 1:
            heap = OneItemHeap()
        elif n_neighbors >= 5:
            heap = MaxHeap()
        else:
            heap = PriorityQueue()
        heap.init(dist_ptr, idx_ptr, n_neighbors)

        # TODO: this loop could be sped up with some cdefs to avoid
        # creating all the Xi sub-arrays
        for i, Xi in enumerate(X):
            pt = <DTYPE_t*>Xi.data
            reduced_dist_LB = self.calc_reduced_dist_LB(pt, node_centroid,
                                                        node_info.radius,
                                                        n_features)
            self.query_one_(0, pt, n_neighbors,
                            dist_ptr, idx_ptr, reduced_dist_LB, heap)

            for i from 0 <= i < k:
                dist_ptr[i] = self.dm.reduced_to_dist(dist_ptr[i],
                                                      &self.dm.params)

            # if max-heap is used, results must be sorted
            if heap.needs_final_sort():
                sort_dist_idx(dist_ptr, idx_ptr, n_neighbors)

            dist_ptr += n_neighbors
            idx_ptr += n_neighbors

        # deflatten results
        if return_distance:
            return (distances.reshape((orig_shape[:-1]) + (k,)),
                    idx_array.reshape((orig_shape[:-1]) + (k,)))
        else:
            return idx_array.reshape((orig_shape[:-1]) + (k,))

    def query_radius(self, X, r, return_distance=False,
                     count_only=False, sort_results=False):
        """
        query_radius(self, X, r, count_only = False):

        query the Ball Tree for neighbors within a ball of size r

        Parameters
        ----------
        X : array-like, last dimension self.dim
            An array of points to query
        r : distance within which neighbors are returned
            r can be a single value, or an array of values of shape
            x.shape[:-1] if different radii are desired for each point.
        return_distance : boolean (default = False)
            if True,  return distances to neighbors of each point
            if False, return only neighbors
            Note that unlike BallTree.query(), setting return_distance=True
            adds to the computation time.  Not all distances need to be
            calculated explicitly for return_distance=False.  Results are
            not sorted by default: see ``sort_results`` keyword.
        count_only : boolean (default = False)
            if True,  return only the count of points within distance r
            if False, return the indices of all points within distance r
            If return_distance==True, setting count_only=True will
            result in an error.
        sort_results : boolean (default = False)
            if True, the distances and indices will be sorted before being
            returned.  If False, the results will not be sorted.  If
            return_distance == False, setting sort_results = True will
            result in an error.

        Returns
        -------
        count       : if count_only == True
        ind         : if count_only == False and return_distance == False
        (ind, dist) : if count_only == False and return_distance == True

        count : array of integers, shape = X.shape[:-1]
            each entry gives the number of neighbors within
            a distance r of the corresponding point.

        ind : array of objects, shape = X.shape[:-1]
            each element is a numpy integer array listing the indices of
            neighbors of the corresponding point.  Note that unlike
            the results of BallTree.query(), the returned neighbors
            are not sorted by distance

        dist : array of objects, shape = X.shape[:-1]
            each element is a numpy double array
            listing the distances corresponding to indices in i.

        Examples
        --------
        Query for neighbors in a given radius

        # >>> import numpy as np
        # >>> np.random.seed(0)
        # >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        # >>> ball_tree = BallTree(X, leaf_size=2)
        # >>> print ball_tree.query_radius(X[0], r=0.3, count_only=True)
        # 3
        # >>> ind = ball_tree.query_radius(X[0], r=0.3)
        # >>> print ind  # indices of neighbors within distance 0.3
        # [3 0 1]
        """
        if count_only and return_distance:
            raise ValueError("count_only and return_distance "
                             "cannot both be true")

        if sort_results and not return_distance:
            raise ValueError("return_distance must be True if sort_distances "
                             "is True")

        cdef np.ndarray idx_array, idx_array_i, distances, distances_i
        cdef np.ndarray pt, count
        cdef ITYPE_t count_i

        # prepare X for query
        X = array2d(X, dtype=DTYPE, order='C')
        if X.shape[-1] != self.data.shape[1]:
            raise ValueError("query data dimension must match BallTree "
                             "data dimension")

        # prepare r for query
        r = np.asarray(r, dtype=DTYPE, order='C')
        r = np.atleast_1d(r)
        if r.shape == (1,):
            r = r[0] * np.ones(X.shape[:-1], dtype=np.double)
        else:
            if r.shape != X.shape[:-1]:
                raise ValueError("r must be broadcastable to X.shape")

        # flatten X and r for iteration
        orig_shape = X.shape
        X = X.reshape((-1, X.shape[-1]))
        r = r.reshape(-1)

        cdef stack node_stack
        stack_create(&node_stack, self.n_levels + 1)

        if count_only:
            count = np.zeros(X.shape[0], ITYPE)

            #TODO: avoid enumerate and repeated allocation of pt slice
            for pt_idx, pt in enumerate(X):
                count[pt_idx] = self.query_radius_count_(<DTYPE_t*>pt.data,
                                                         r[pt_idx],
                                                         &node_stack)
        elif not return_distance:
            idx_array = np.empty(X.shape[0], dtype='object')
            idx_array_i = np.empty(self.data.shape[0], dtype=ITYPE)

            #TODO: avoid enumerate and repeated allocation of pt slice
            for pt_idx, pt in enumerate(X):
                count_i = self.query_radius_idx_only_(
                    <DTYPE_t*>pt.data,
                    r[pt_idx],
                    <ITYPE_t*>idx_array_i.data,
                    &node_stack)
                idx_array[pt_idx] = idx_array_i[:count_i].copy()

        else:
            idx_array = np.empty(X.shape[0], dtype='object')
            distances = np.empty(X.shape[0], dtype='object')
            idx_array_i = np.empty(self.data.shape[0], dtype=ITYPE)
            distances_i = np.empty(self.data.shape[0], dtype=DTYPE)

            #TODO: avoid enumerate and repeated allocation of pt slice
            for pt_idx, pt in enumerate(X):
                count_i = self.query_radius_distances_(
                    <DTYPE_t*>pt.data,
                    r[pt_idx],
                    <ITYPE_t*>idx_array_i.data,
                    <DTYPE_t*>distances_i.data,
                    &node_stack)
                if sort_results:
                    sort_dist_idx(<DTYPE_t*>distances_i.data,
                                  <ITYPE_t*>idx_array_i.data,
                                  count_i)

                idx_array[pt_idx] = idx_array_i[:count_i].copy()
                distances[pt_idx] = distances_i[:count_i].copy()

        stack_destroy(&node_stack)

        # deflatten results
        if count_only:
            return count.reshape(orig_shape[:-1])
        elif return_distance:
            return (idx_array.reshape(orig_shape[:-1]),
                    distances.reshape(orig_shape[:-1]))
        else:
            return idx_array.reshape(orig_shape[:-1])

    @cython.cdivision(True)
    cdef void build_tree_(BallTree self):
        cdef DTYPE_t* data = <DTYPE_t*> self.data.data
        cdef ITYPE_t* idx_array = <ITYPE_t*> self.idx_array.data
        cdef DTYPE_t* node_centroid_arr = <DTYPE_t*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data

        cdef ITYPE_t n_samples = self.data.shape[0]
        cdef ITYPE_t n_features = self.data.shape[1]

        cdef ITYPE_t idx_start, idx_end, n_points
        cdef DTYPE_t radius
        cdef ITYPE_t i, i_node, i_parent

        cdef DTYPE_t* centroid = node_centroid_arr
        cdef NodeInfo* node_info = node_info_arr
        cdef NodeInfo* parent_info
        cdef DTYPE_t* point

        #------------------------------------------------------------
        # take care of the root node
        node_info.idx_start = 0
        node_info.idx_end = n_samples
        n_points = n_samples

        # determine Node centroid
        compute_centroid(centroid, data, idx_array,
                         n_features, n_samples)

        # determine Node radius
        radius = 0
        for i from node_info.idx_start <= i < node_info.idx_end:
            radius = fmax(radius,
                          self.dm.reduced_dfunc(
                                  centroid, data + n_features * idx_array[i],
                                  n_features, &self.dm.params, -1, -1))
        node_info.radius = self.dm.reduced_to_dist(radius, &self.dm.params)

        # check if this is a leaf
        if self.n_nodes == 1:
            node_info.is_leaf = 1

        else:
            node_info.is_leaf = 0

            # find dimension with largest spread
            i_max = find_split_dim(data, idx_array + node_info.idx_start,
                                   n_features, n_points)

            # sort idx_array along this dimension
            partition_indices(data,
                              idx_array + node_info.idx_start,
                              i_max,
                              n_points / 2,
                              n_features,
                              n_points)

        #------------------------------------------------------------
        # cycle through all child nodes
        for i_node from 1 <= i_node < self.n_nodes:
            i_parent = (i_node - 1) / 2
            parent_info = node_info_arr + i_parent

            node_info = node_info_arr + i_node

            if parent_info.is_leaf:
                # sanity check.  This should never happen
                raise ValueError("Fatal: parent is a leaf. "
                                 "Memory layout is flawed")

            if i_node < self.n_nodes / 2:
                node_info.is_leaf = 0
            else:
                node_info.is_leaf = 1

            centroid = node_centroid_arr + i_node * n_features

            # find indices for this node
            idx_start = parent_info.idx_start
            idx_end = parent_info.idx_end

            if i_node % 2 == 1:
                idx_start = (idx_start + idx_end) / 2
            else:
                idx_end = (idx_start + idx_end) / 2

            node_info.idx_start = idx_start
            node_info.idx_end = idx_end

            n_points = idx_end - idx_start

            if n_points == 0:
                # sanity check: this should never happen
                raise ValueError("Fatal: zero-sized node. "
                                 "Memory layout is flawed")

            elif n_points == 1:
                #copy this point to centroid
                copy_array(centroid,
                           data + idx_array[idx_start] * n_features,
                           n_features)

                #store radius in array
                node_info.radius = 0

                #is a leaf
                node_info.is_leaf = 1

            else:
                # determine Node centroid
                compute_centroid(centroid, data, idx_array + idx_start,
                                 n_features, n_points)

                # determine Node radius
                radius = 0
                for i from idx_start <= i < idx_end:
                    radius = fmax(radius,
                                  self.dm.reduced_dfunc(
                        centroid, data + n_features * idx_array[i],
                        n_features, &self.dm.params, -1, -1))
                node_info.radius = self.dm.reduced_to_dist(radius,
                                                           &self.dm.params)

                if not node_info.is_leaf:
                    # find dimension with largest spread
                    i_max = find_split_dim(data, idx_array + idx_start,
                                           n_features, n_points)

                    # sort indices along this dimension
                    partition_indices(data,
                                      idx_array + idx_start,
                                      i_max,
                                      n_points / 2,
                                      n_features,
                                      n_points)

    cdef void query_one_(BallTree self,
                         ITYPE_t i_node,
                         DTYPE_t* pt,
                         ITYPE_t k,
                         DTYPE_t* near_set_dist,
                         ITYPE_t* near_set_indx,
                         DTYPE_t reduced_dist_LB,
                         HeapBase heap):
        cdef DTYPE_t* data = <DTYPE_t*> self.data.data
        cdef ITYPE_t* idx_array = <ITYPE_t*> self.idx_array.data
        cdef DTYPE_t* node_centroid_arr = <DTYPE_t*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data

        cdef ITYPE_t n_features = self.data.shape[1]

        cdef DTYPE_t dist_pt, reduced_dist_LB_1, reduced_dist_LB_2
        cdef ITYPE_t i, i1, i2

        # get the node info for this node
        cdef NodeInfo* node_info = node_info_arr + i_node

        # set the values in the heap
        heap.val = near_set_dist
        heap.idx = near_set_indx

        #------------------------------------------------------------
        # Case 1: query point is outside node radius trim the query
        if reduced_dist_LB > heap.largest():
            pass

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node_info.is_leaf:
            for i from node_info.idx_start <= i < node_info.idx_end:
                dist_pt = self.dm.reduced_dfunc(
                    pt, data + n_features * idx_array[i], n_features,
                    &self.dm.params, -1, -1)

                if dist_pt < heap.largest():
                    heap.insert(dist_pt, idx_array[i])

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the one whose centroid is closest
        else:
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            reduced_dist_LB_1 = self.calc_reduced_dist_LB(
                pt, (node_centroid_arr + i1 * n_features),
                node_info_arr[i1].radius, n_features)
            reduced_dist_LB_2 = self.calc_reduced_dist_LB(
                pt, (node_centroid_arr + i2 * n_features),
                node_info_arr[i2].radius, n_features)

            # recursively call query_one
            if reduced_dist_LB_1 <= reduced_dist_LB_2:
                self.query_one_(i1, pt, k, near_set_dist, near_set_indx,
                                reduced_dist_LB_1, heap)
                self.query_one_(i2, pt, k, near_set_dist, near_set_indx,
                                reduced_dist_LB_2, heap)
            else:
                self.query_one_(i2, pt, k, near_set_dist, near_set_indx,
                                reduced_dist_LB_2, heap)
                self.query_one_(i1, pt, k, near_set_dist, near_set_indx,
                                reduced_dist_LB_1, heap)


    cdef ITYPE_t query_radius_count_(BallTree self,
                                     DTYPE_t* pt, DTYPE_t r,
                                     stack* node_stack):
        cdef DTYPE_t* data = <DTYPE_t*> self.data.data
        cdef ITYPE_t* idx_array = <ITYPE_t*> self.idx_array.data
        cdef DTYPE_t* node_centroid_arr = <DTYPE_t*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t i, i_node
        cdef ITYPE_t count = 0
        cdef DTYPE_t reduced_r = self.dm.dist_to_reduced(r, &self.dm.params)
        cdef DTYPE_t dist_pt

        cdef stack_item item

        item.i_node = 0
        stack_push(node_stack, item)

        while(node_stack.n > 0):
            item = stack_pop(node_stack)
            i_node = item.i_node
            node_info = node_info_arr + i_node

            dist_pt = self.dm.dfunc(
                            pt, node_centroid_arr + n_features * i_node,
                            n_features, &self.dm.params, -1, -1)

            #------------------------------------------------------------
            # Case 1: all node points are outside distance r.
            #         prune this branch.
            if dist_pt - node_info.radius > r:
                continue

            #------------------------------------------------------------
            # Case 2: all node points are within distance r
            #         add all points
            elif dist_pt + node_info.radius < r:
                count += (node_info.idx_end - node_info.idx_start)

            #------------------------------------------------------------
            # Case 3: this is a leaf node.  Go through all points to
            #         determine if they fall within radius
            elif node_info.is_leaf:
                for i from node_info.idx_start <= i < node_info.idx_end:
                    dist_pt = self.dm.reduced_dfunc(
                            pt, data + n_features * idx_array[i],
                            n_features, &self.dm.params, -1, -1)
                    if dist_pt <= reduced_r:
                        count += 1

            #------------------------------------------------------------
            # Case 4: Node is not a leaf.  Recursively query subnodes
            else:
                item.i_node = 2 * i_node + 1
                stack_push(node_stack, item)

                item.i_node = i = 2 * i_node + 2
                stack_push(node_stack, item)

        return count

    cdef ITYPE_t query_radius_idx_only_(BallTree self,
                                        DTYPE_t* pt, DTYPE_t r,
                                        ITYPE_t* indices,
                                        stack* node_stack):
        cdef DTYPE_t* data = <DTYPE_t*> self.data.data
        cdef ITYPE_t* idx_array = <ITYPE_t*> self.idx_array.data
        cdef DTYPE_t* node_centroid_arr = <DTYPE_t*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t i, i_node
        cdef ITYPE_t idx_i = 0
        cdef DTYPE_t reduced_r = self.dm.dist_to_reduced(r, &self.dm.params)
        cdef DTYPE_t dist_pt

        cdef stack_item item

        item.i_node = 0
        stack_push(node_stack, item)

        while(node_stack.n > 0):
            item = stack_pop(node_stack)
            i_node = item.i_node
            node_info = node_info_arr + i_node

            dist_pt = self.dm.dfunc(pt,
                                    node_centroid_arr + n_features * i_node,
                                    n_features, &self.dm.params, -1, -1)

            #------------------------------------------------------------
            # Case 1: all node points are outside distance r.
            #         prune this branch.
            if dist_pt - node_info.radius > r:
                continue

            #------------------------------------------------------------
            # Case 2: all node points are within distance r
            #         add all points
            elif dist_pt + node_info.radius < r:
                for i from node_info.idx_start <= i < node_info.idx_end:
                    indices[idx_i] = idx_array[i]
                    idx_i += 1

            #------------------------------------------------------------
            # Case 3: this is a leaf node.  Go through all points to
            #         determine if they fall within radius
            elif node_info.is_leaf:
                for i from node_info.idx_start <= i < node_info.idx_end:
                    dist_pt = self.dm.reduced_dfunc(
                            pt, data + n_features * idx_array[i], n_features,
                            &self.dm.params, -1, -1)
                    if dist_pt <= reduced_r:
                        indices[idx_i] = idx_array[i]
                        idx_i += 1

            #------------------------------------------------------------
            # Case 4: Node is not a leaf.  Recursively query subnodes
            else:
                item.i_node = 2 * i_node + 1
                stack_push(node_stack, item)

                item.i_node = i = 2 * i_node + 2
                stack_push(node_stack, item)

        return idx_i

    cdef ITYPE_t query_radius_distances_(BallTree self,
                                         DTYPE_t* pt, DTYPE_t r,
                                         ITYPE_t* indices,
                                         DTYPE_t* distances,
                                         stack* node_stack):
        cdef DTYPE_t* data = <DTYPE_t*> self.data.data
        cdef ITYPE_t* idx_array = <ITYPE_t*> self.idx_array.data
        cdef DTYPE_t* node_centroid_arr = <DTYPE_t*>self.node_centroid_arr.data
        cdef NodeInfo* node_info_arr = <NodeInfo*> self.node_info_arr.data
        cdef NodeInfo* node_info = node_info_arr

        cdef ITYPE_t n_features = self.data.shape[1]
        cdef ITYPE_t i, i_node
        cdef ITYPE_t idx_i = 0
        cdef DTYPE_t reduced_r = self.dm.dist_to_reduced(r, &self.dm.params)
        cdef DTYPE_t dist_pt

        cdef stack_item item

        item.i_node = 0
        stack_push(node_stack, item)

        while(node_stack.n > 0):
            item = stack_pop(node_stack)
            i_node = item.i_node
            node_info = node_info_arr + i_node

            dist_pt = self.dm.dfunc(pt,
                                    node_centroid_arr + n_features * i_node,
                                    n_features, &self.dm.params, -1, -1)

            #------------------------------------------------------------
            # Case 1: all node points are outside distance r.
            #         prune this branch.
            if dist_pt - node_info.radius > r:
                continue

            #------------------------------------------------------------
            # Case 2: all node points are within distance r
            #         add all points
            elif dist_pt + node_info.radius < r:
                for i from node_info.idx_start <= i < node_info.idx_end:
                    dist_pt = self.dm.dfunc(pt,
                                            data + n_features * idx_array[i],
                                            n_features, &self.dm.params, -1, -1)
                    indices[idx_i] = idx_array[i]
                    distances[idx_i] = dist_pt
                    idx_i += 1

            #------------------------------------------------------------
            # Case 3: this is a leaf node.  Go through all points to
            #         determine if they fall within radius
            elif node_info.is_leaf:
                for i from node_info.idx_start <= i < node_info.idx_end:
                    dist_pt = self.dm.reduced_dfunc(
                            pt, data + n_features * idx_array[i],
                            n_features, &self.dm.params, -1, -1)
                    if dist_pt <= reduced_r:
                        indices[idx_i] = idx_array[i]
                        distances[idx_i] = self.dm.reduced_to_dist(
                            dist_pt, &self.dm.params)
                        idx_i += 1

            #------------------------------------------------------------
            # Case 4: Node is not a leaf.  Recursively query subnodes
            else:
                item.i_node = 2 * i_node + 1
                stack_push(node_stack, item)

                item.i_node = i = 2 * i_node + 2
                stack_push(node_stack, item)

        return idx_i

    

    ########################################################################
    # calc_dist_LB
    # calc_reduced_dist_LB
    #  These calculates the lower-bound distances between a point and a node
    cdef inline DTYPE_t calc_dist_LB(BallTree self,
                                     DTYPE_t* pt,
                                     DTYPE_t* centroid,
                                     DTYPE_t radius,
                                     ITYPE_t n_features):
        return fmax(
            0, (self.dm.dfunc(pt, centroid, n_features,
                              &self.dm.params, -1, -1)
                - radius))

    cdef inline DTYPE_t calc_reduced_dist_LB(BallTree self,
                                             DTYPE_t* pt,
                                             DTYPE_t* centroid,
                                             DTYPE_t radius,
                                             ITYPE_t n_features):
        return self.dm.dist_to_reduced(
            fmax(0, (self.dm.dfunc(pt, centroid, n_features,
                                   &self.dm.params, -1, -1)
                     - radius)),
            &self.dm.params)


######################################################################
# Helper functions for building and querying
#
@cython.profile(False)
cdef inline void copy_array(DTYPE_t* x, DTYPE_t* y, ITYPE_t n):
    # copy array y into array x
    cdef ITYPE_t i
    for i from 0 <= i < n:
        x[i] = y[i]


@cython.cdivision(True)
cdef void compute_centroid(DTYPE_t* centroid,
                           DTYPE_t* data,
                           ITYPE_t* node_indices,
                           ITYPE_t n_features,
                           ITYPE_t n_points):
    # `centroid` points to an array of length n_features
    # `data` points to an array of length n_samples * n_features
    # `node_indices` = idx_array + idx_start
    cdef DTYPE_t *this_pt
    cdef ITYPE_t i, j

    for j from 0 <= j < n_features:
        centroid[j] = 0

    for i from 0 <= i < n_points:
        this_pt = data + n_features * node_indices[i]
        for j from 0 <= j < n_features:
            centroid[j] += this_pt[j]

    for j from 0 <= j < n_features:
        centroid[j] /= n_points


cdef ITYPE_t find_split_dim(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
    # this computes the following
    # j_max = np.argmax(np.max(data, 0) - np.min(data, 0))
    cdef DTYPE_t min_val, max_val, val, spread, max_spread
    cdef ITYPE_t i, j, j_max

    j_max = 0
    max_spread = 0

    for j from 0 <= j < n_features:
        max_val = data[node_indices[0] * n_features + j]
        min_val = max_val
        for i from 1 <= i < n_points:
            val = data[node_indices[i] * n_features + j]
            max_val = fmax(max_val, val)
            min_val = fmin(min_val, val)
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max


@cython.profile(False)
cdef inline void iswap(ITYPE_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef ITYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


@cython.profile(False)
cdef inline void dswap(DTYPE_t* arr, ITYPE_t i1, ITYPE_t i2):
    cdef DTYPE_t tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


cdef void partition_indices(DTYPE_t* data,
                            ITYPE_t* node_indices,
                            ITYPE_t split_dim,
                            ITYPE_t split_index,
                            ITYPE_t n_features,
                            ITYPE_t n_points):
    # partition_indices will modify the array node_indices between
    # indices 0 and n_points.  Upon return (assuming numpy-style slicing)
    #   data[node_indices[0:split_index], split_dim]
    #     <= data[node_indices[split_index], split_dim]
    # and
    #   data[node_indices[split_index], split_dim]
    #     <= data[node_indices[split_index:n_points], split_dim]
    # will hold.  The algorithm amounts to a partial quicksort
    cdef ITYPE_t left, right, midindex, i
    cdef DTYPE_t d1, d2
    left = 0
    right = n_points - 1

    while True:
        midindex = left
        for i from left <= i < right:
            d1 = data[node_indices[i] * n_features + split_dim]
            d2 = data[node_indices[right] * n_features + split_dim]
            if d1 < d2:
                iswap(node_indices, i, midindex)
                midindex += 1
        iswap(node_indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1

######################################################################
# sort_dist_idx :
#  this is a recursive quicksort implementation which sorts `dist` and
#  simultaneously performs the same swaps on `idx`.
cdef void sort_dist_idx(DTYPE_t* dist, ITYPE_t* idx, ITYPE_t k):
    cdef ITYPE_t pivot_idx, store_idx, i
    cdef DTYPE_t pivot_val

    if k > 1:
        #-- determine pivot -----------
        pivot_idx = k / 2
        pivot_val = dist[pivot_idx]
        store_idx = 0
                         
        dswap(dist, pivot_idx, k - 1)
        iswap(idx, pivot_idx, k - 1)

        for i from 0 <= i < k - 1:
            if dist[i] < pivot_val:
                dswap(dist, i, store_idx)
                iswap(idx, i, store_idx)
                store_idx += 1
        dswap(dist, store_idx, k - 1)
        iswap(idx, store_idx, k - 1)
        pivot_idx = store_idx
        #------------------------------

        sort_dist_idx(dist, idx, pivot_idx)

        sort_dist_idx(dist + pivot_idx + 1,
                      idx + pivot_idx + 1,
                      k - pivot_idx - 1)


######################################################################
# heap implementation
#
# We use an inheritance structure to allow multiple implementations with
# the same interface.  As long as each derived class only overloads functions
# in the base class rather than defining new functions, this will allow
# very fast execution.

#----------------------------------------------------------------------
# Heap base class
cdef class HeapBase:
    cdef DTYPE_t* val
    cdef ITYPE_t* idx
    cdef ITYPE_t size

    cdef int needs_final_sort(self):
        return 0

    cdef void init(self, DTYPE_t* val, ITYPE_t* idx, ITYPE_t size):
        self.val = val
        self.idx = idx
        self.size = size

    cdef DTYPE_t largest(self):
        return 0.0

    cdef ITYPE_t idx_largest(self):
        return 0

    cdef void insert(self, DTYPE_t val, ITYPE_t i_val):
        pass

#----------------------------------------------------------------------
# MaxHeap
#
#  This is a basic implementation of a fixed-size binary max-heap.
#  It can be used in place of priority_queue to keep track of the
#  k-nearest neighbors in a query.  The implementation is faster than
#  priority_queue for a very large number of neighbors (k > 50 or so).
#  The implementation is slower than priority_queue for fewer neighbors.
#  The other disadvantage is that for max_heap, the indices/distances must
#  be sorted upon completion of the query.  In priority_queue, the indices
#  and distances are sorted without an extra call.
#
#  The root node is at heap[0].  The two child nodes of node i are at
#  (2 * i + 1) and (2 * i + 2).
#  The parent node of node i is node floor((i-1)/2).  Node 0 has no parent.
#  A max heap has (heap[i] >= heap[2 * i + 1]) and (heap[i] >= heap[2 * i + 2])
#  for all valid indices.
#
#  In this implementation, an empty heap should be full of infinities
#
cdef class MaxHeap(HeapBase):
    cdef int needs_final_sort(self):
        return 1

    cdef DTYPE_t largest(self):
        return self.val[0]

    cdef ITYPE_t idx_largest(self):
        return self.idx[0]

    cdef void insert(self, DTYPE_t val, ITYPE_t i_val):
        cdef ITYPE_t i, ic1, ic2, i_tmp
        cdef DTYPE_t d_tmp

        # check if val should be in heap
        if val > self.val[0]:
            return

        # insert val at position zero
        self.val[0] = val
        self.idx[0] = i_val

        #descend the heap, swapping values until the max heap criterion is met
        i = 0
        while 1:
            ic1 = 2 * i + 1
            ic2 = ic1 + 1

            if ic1 >= self.size:
                break
            elif ic2 >= self.size:
                if self.val[ic1] > val:
                    i_swap = ic1
                else:
                    break
            elif self.val[ic1] >= self.val[ic2]:
                if val < self.val[ic1]:
                    i_swap = ic1
                else:
                    break
            else:
                if val < self.val[ic2]:
                    i_swap = ic2
                else:
                    break

            self.val[i] = self.val[i_swap]
            self.idx[i] = self.idx[i_swap]

            i = i_swap

        self.val[i] = val
        self.idx[i] = i_val


#----------------------------------------------------------------------
# Priority Queue Implementation
#
#  This is used to keep track of the neighbors as they are found.
#  It keeps the list of neighbors sorted, and inserts each new item
#  into the list.  In this fixed-size implementation, empty elements
#  are represented by infinities.
cdef class PriorityQueue(HeapBase):
    cdef int needs_final_sort(self):
        return 0

    cdef DTYPE_t largest(self):
        return self.val[self.size - 1]

    cdef ITYPE_t pqueue_idx_largest(self):
        return self.idx[self.size - 1]

    cdef void insert(self, DTYPE_t val, ITYPE_t i_val):
        cdef ITYPE_t i_lower = 0
        cdef ITYPE_t i_upper = self.size - 1
        cdef ITYPE_t i, i_mid

        if val >= self.val[i_upper]:
            return
        elif val <= self.val[i_lower]:
            i_mid = i_lower
        else:
            while True:
                if (i_upper - i_lower) < 2:
                    i_mid = i_lower + 1
                    break
                else:
                    i_mid = (i_lower + i_upper) / 2

                if i_mid == i_lower:
                    i_mid += 1
                    break

                if val >= self.val[i_mid]:
                    i_lower = i_mid
                else:
                    i_upper = i_mid

        for i from self.size > i > i_mid:
            self.val[i] = self.val[i - 1]
            self.idx[i] = self.idx[i - 1]

        self.val[i_mid] = val
        self.idx[i_mid] = i_val


#----------------------------------------------------------------------
# One item implementation
#
# In the common case of a single neighbor, we can use a simple "heap" of
# one item which is more efficient than either of the above options
cdef class OneItemHeap(HeapBase):
    cdef int needs_final_sort(self):
        return 0

    cdef DTYPE_t largest(self):
        return self.val[0]

    cdef ITYPE_t pqueue_idx_largest(self):
        return self.idx[0]

    cdef void insert(self, DTYPE_t val, ITYPE_t i_val):
        if val < self.val[0]:
            self.val[0] = val
            self.idx[0] = i_val
