import warnings

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fabs, fmax, sqrt, pow

cdef extern from "stdlib.h":
    int strcmp(char *a, char *b)

cdef extern from "arrayobject.h":
    object PyArray_SimpleNewFromData(int nd, np.npy_intp* dims, 
                                     int typenum, void* data)
np.import_array()  # required in order to use C-API

# python data types (corresponding C-types are in pxd file)
DTYPE = np.float64

# TODO:
#  Functionality:
#   - add `override_precomputed` flag on distance functions
#
#  Speed:
#   - use blas for computations where appropriate
#   - boolean functions are slow: how do we access fast C boolean operations?
#   - use @cython.cdivision(True) where applicable
#   - enable fast euclidean distances using (x-y)^2 = x^2 + y^2 - 2xy
#     and 'precomputed norms' flag
#
#  Documentation:
#   - documentation of metrics
#   - double-check consistency with sklearn.metrics & scipy.spatial.distance
#
#  Future Functionality:
#   - make cdist/pdist work with fortran arrays (see note below)
#   - make cdist/pdist work with csr matrices.  This will require writing
#     a new form of each distance function which accepts csr input.
#   - implement KD tree based on this (?)
#   - cover tree as well (?)
#   - templating?  this would be a great candidate to try out cython templates.
#

# One idea:
#  to save on memory, we could define general distance functions with
#   the signature
#  dfunc(DTYPE_t* x1, DTYPE_t* x2, Py_ssize_t n,
#        Py_ssize_t rowstride1, Py_ssize_t colstride1,
#        Py_ssize_t rowstride2, Py_ssize_t colstride2,
#        Py_ssize_t rowindex1,  Py_ssize_t rowindex2,
#        dist_params* params)
#
#  This would allow arbitrary numpy arrays to be used by the function,
#   but would slightly slow down computation.


###############################################################################
# Helper functions
cdef np.ndarray _norms(np.ndarray X):
    return np.sqrt(np.asarray((X ** 2).sum(1), dtype=DTYPE, order='C'))

cdef np.ndarray _centered(np.ndarray X):
    return X - X.mean(1).reshape((-1, 1))

cdef inline np.ndarray _buffer_to_ndarray(DTYPE_t* x, np.npy_intp n):
    # Wrap a memory buffer with an ndarray.  Warning: this is not robust.
    # In particular, if x is deallocated before the returned array goes
    # out of scope, this could SegFault.

    # if we know what n is beforehand, we can simply call
    # (in newer cython versions)
    #return np.asarray(<double[:100]> x)

    # Note: this Segfaults unless np.import_array() is called above
    return PyArray_SimpleNewFromData(1, &n, DTYPECODE, <void*>x)


###############################################################################
#Here we define the various distance functions
#
# Distance functions have the following call signature
#
# distance(DTYPE_t* x1, DTYPE_t* x1, Py_ssize_t n,
#          dist_params* params,
#          Py_ssize_t rowindex1, Py_ssize_t rowindex2)
#
#     Parameters
#     ----------
#     x1, x2 : double*
#         pointers to data arrays
#     n : integer
#         length of vector
#     params : structure
#         the parameter structure contains various parameters that define
#         the distance metric, or aid in faster computation.
#     rowindex1, rowindex2 : integers
#         these define the offsets for precomputed values
#         if either is negative, then no precomputed value will be used.
#    
#     Returns
#     -------
#     D : double
#         distance between v1 and v2
#
###############################################################################


cdef DTYPE_t euclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d
    
    return sqrt(res)


cdef DTYPE_t manhattan_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t res = 0
    
    for i from 0 <= i < n:
        res += fabs(x1[i] - x2[i])

    return res


cdef DTYPE_t chebyshev_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t res = 0
    
    for i from 0 <= i < n:
        res = fmax(res, fabs(x1[i] - x2[i]))

    return res


cdef DTYPE_t minkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += pow(d, params.minkowski.p)

    return pow(res, 1. / params.minkowski.p)


cdef DTYPE_t pminkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += pow(d, params.minkowski.p)

    return res


cdef DTYPE_t wminkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0
    
    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += pow(params.minkowski.w[i] * d, params.minkowski.p)

    return pow(res, 1. / params.minkowski.p)


cdef DTYPE_t pwminkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  Py_ssize_t n, dist_params* params,
                                  Py_ssize_t rowindex1,
                                  Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0
    
    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += pow(params.minkowski.w[i] * d, params.minkowski.p)

    return res


cdef DTYPE_t mahalanobis_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  Py_ssize_t n, dist_params* params,
                                  Py_ssize_t rowindex1,
                                  Py_ssize_t rowindex2):
    cdef Py_ssize_t i, j
    cdef DTYPE_t d, res = 0

    assert n == params.mahalanobis.n

    # TODO: use blas here
    for i from 0 <= i < n:
        params.mahalanobis.work_buffer[i] = x1[i] - x2[i]

    for i from 0 <= i < n:
        d = 0
        for j from 0 <= j < n:
            d += (params.mahalanobis.VI[i * n + j]
                  * params.mahalanobis.work_buffer[j])
        res += d * params.mahalanobis.work_buffer[i]

    return sqrt(res)


cdef DTYPE_t sqmahalanobis_distance(DTYPE_t* x1, DTYPE_t* x2,
                                    Py_ssize_t n, dist_params* params,
                                    Py_ssize_t rowindex1,
                                    Py_ssize_t rowindex2):
    cdef Py_ssize_t i, j
    cdef DTYPE_t d, res = 0

    assert n == params.mahalanobis.n

    # TODO: use blas here
    for i from 0 <= i < n:
        params.mahalanobis.work_buffer[i] = x1[i] - x2[i]

    for i from 0 <= i < n:
        d = 0
        for j from 0 <= j < n:
            d += (params.mahalanobis.VI[i * n + j]
                  * params.mahalanobis.work_buffer[j])
        res += d * params.mahalanobis.work_buffer[i]

    return res


cdef DTYPE_t seuclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0
    
    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d / params.seuclidean.V[i]
    
    return sqrt(res)


cdef DTYPE_t sqseuclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0
    
    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d / params.seuclidean.V[i]
    
    return res


cdef DTYPE_t sqeuclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  Py_ssize_t n, dist_params* params,
                                  Py_ssize_t rowindex1,
                                  Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d
    
    return res


cdef DTYPE_t cosine_distance(DTYPE_t* x1, DTYPE_t* x2,
                             Py_ssize_t n, dist_params* params,
                             Py_ssize_t rowindex1,
                             Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t x1nrm = 0, x2nrm = 0, x1Tx2 = 0, normalization = 0

    cdef int precomputed1 = (rowindex1 >= 0)
    cdef int precomputed2 = (rowindex2 >= 0)

    # TODO: use blas here
    if params.cosine.precomputed_norms and precomputed1 and precomputed2:
        for i from 0 <= i < n:
            x1Tx2 += x1[i] * x2[i]

        normalization = params.cosine.norms1[rowindex1]
        normalization *= params.cosine.norms2[rowindex2]
    
    else:
        for i from 0 <= i < n:
            x1nrm += x1[i] * x1[i]
            x2nrm += x2[i] * x2[i]
            x1Tx2 += x1[i] * x2[i]

        normalization = sqrt(x1nrm * x2nrm)

    return 1.0 - (x1Tx2) / normalization


cdef DTYPE_t correlation_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  Py_ssize_t n, dist_params* params,
                                  Py_ssize_t rowindex1,
                                  Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t mu1 = 0, mu2 = 0, x1nrm = 0, x2nrm = 0, x1Tx2 = 0
    cdef DTYPE_t normalization

    cdef DTYPE_t tmp1, tmp2

    cdef int precomputed1 = (rowindex1 >= 0)
    cdef int precomputed2 = (rowindex2 >= 0)

    # TODO : use blas here
    if params.correlation.precomputed_data and precomputed1 and precomputed2:
        x1 = params.correlation.x1 + rowindex1 * n
        x2 = params.correlation.x2 + rowindex2 * n

        for i from 0 <= i < n:
            x1Tx2 += x1[i] * x2[i]

        normalization = params.correlation.norms1[rowindex1]
        normalization *= params.correlation.norms2[rowindex2]

    else:
        for i from 0 <= i < n:
            mu1 += x1[i]
            mu2 += x2[i]
        mu1 /= n
        mu2 /= n

        for i from 0 <= i < n:
            tmp1 = x1[i] - mu1
            tmp2 = x2[i] - mu2
            x1nrm += tmp1 * tmp1
            x2nrm += tmp2 * tmp2
            x1Tx2 += tmp1 * tmp2

        normalization = sqrt(x1nrm * x2nrm)

    return 1. - x1Tx2 / normalization


cdef DTYPE_t hamming_distance(DTYPE_t* x1, DTYPE_t* x2,
                              Py_ssize_t n, dist_params* params,
                              Py_ssize_t rowindex1,
                              Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef int n_disagree = 0

    for i from 0 <= i < n:
        n_disagree += (x1[i] != x2[i])

    return n_disagree * 1. / n


cdef DTYPE_t jaccard_distance(DTYPE_t* x1, DTYPE_t* x2,
                              Py_ssize_t n, dist_params* params,
                              Py_ssize_t rowindex1,
                              Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef int num = 0, denom = 0

    for i from 0 <= i < n:
        if (x1[i] != x2[i]) and (x1[i] != 0 or x2[i] != 0):
            num += 1
        if x1[i] != 0 or x2[i] != 0:
            denom += 1

    return num * 1. / denom


cdef DTYPE_t canberra_distance(DTYPE_t* x1, DTYPE_t* x2,
                               Py_ssize_t n, dist_params* params,
                               Py_ssize_t rowindex1,
                               Py_ssize_t rowindex2):
    cdef DTYPE_t res = 0, denominator
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        denominator = (fabs(x1[i]) + fabs(x2[i]))
        if denominator > 0:
            res += fabs(x1[i] - x2[i]) / denominator

    return res


cdef DTYPE_t braycurtis_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t numerator = 0, denominator = 0

    for i from 0 <= i < n:
        numerator += fabs(x1[i] - x2[i])
        denominator += fabs(x1[i])
        denominator += fabs(x2[i])

    return numerator / denominator

@cython.cdivision(True)
cdef DTYPE_t yule_distance(DTYPE_t* x1, DTYPE_t* x2,
                           Py_ssize_t n, dist_params* params,
                           Py_ssize_t rowindex1,
                           Py_ssize_t rowindex2):
    cdef int TF1, TF2, ntt = 0, nff = 0, ntf = 0, nft = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        nff += (1 - TF1) * (1 - TF2)
        nft += (1 - TF1) * TF2
        ntf += TF1 * (1 - TF2)
        ntt += TF1 * TF2

    return (2. * ntf * nft) / (ntt * nff + ntf * nft)


cdef DTYPE_t matching_distance(DTYPE_t* x1, DTYPE_t* x2,
                               Py_ssize_t n, dist_params* params,
                               Py_ssize_t rowindex1,
                               Py_ssize_t rowindex2):
    cdef int TF1, TF2, n_neq = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        n_neq += (TF1 != TF2)

    return n_neq * 1. / n


cdef DTYPE_t dice_distance(DTYPE_t* x1, DTYPE_t* x2,
                           Py_ssize_t n, dist_params* params,
                           Py_ssize_t rowindex1,
                           Py_ssize_t rowindex2):
    cdef int TF1, TF2, ntt = 0, n_neq = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2
        n_neq += (TF1 != TF2)

    return n_neq * 1. / (2 * ntt + n_neq)


cdef DTYPE_t kulsinski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef int TF1, TF2, ntt = 0, n_neq = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2
        n_neq += (TF1 != TF2)

    return (n_neq - ntt + n) * 1. / (n_neq + n)


cdef DTYPE_t rogerstanimoto_distance(DTYPE_t* x1, DTYPE_t* x2,
                                     Py_ssize_t n, dist_params* params,
                                     Py_ssize_t rowindex1,
                                     Py_ssize_t rowindex2):
    cdef int TF1, TF2, n_neq = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        n_neq += (TF1 != TF2)

    return n_neq * 2. / (n + n_neq)


cdef DTYPE_t russellrao_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef int TF1, TF2, ntt = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2

    return (n - ntt) * 1. / n


cdef DTYPE_t sokalmichener_distance(DTYPE_t* x1, DTYPE_t* x2,
                                    Py_ssize_t n, dist_params* params,
                                    Py_ssize_t rowindex1,
                                    Py_ssize_t rowindex2):
    cdef int TF1, TF2, n_neq = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        n_neq += (TF1 != TF2)

    return n_neq * 2.0 / (n + n_neq)


cdef DTYPE_t sokalsneath_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  Py_ssize_t n, dist_params* params,
                                  Py_ssize_t rowindex1,
                                  Py_ssize_t rowindex2):
    cdef int TF1, TF2, ntt = 0, n_neq = 0
    cdef Py_ssize_t i

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2
        n_neq += (TF1 != TF2)

    return n_neq * 2.0 / (ntt + 2 * n_neq)


cdef DTYPE_t user_distance(DTYPE_t* x1, DTYPE_t* x2,
                           Py_ssize_t n, dist_params* params,
                           Py_ssize_t rowindex1,
                           Py_ssize_t rowindex2):
    cdef np.ndarray y1 = _buffer_to_ndarray(x1, n)
    cdef np.ndarray y2 = _buffer_to_ndarray(x2, n)
    return (<object>(params.user.func))(y1, y2)


######################################################################
# conversions between reduced and standard distances
#
# Motivation
#  for some distances the full computation does not have to be performed
#  in order to compare distances.  For example, with euclidean distance,
#  to find out if x1 or x2 is closer to y, one only needs to compare
#  sum((x1 - y) ** 2) and sum((x2 - y) ** 2).  That is, the square root
#  does not need to be performed.  In order to take advantage of this
#  within BallTree, we need a way of recognizing this for various metrics
#  and converting between the distances.
#

cdef inline DTYPE_t no_conversion(DTYPE_t x, dist_params* params):
    return x


cdef inline DTYPE_t euclidean_from_reduced(DTYPE_t x, dist_params* params):
    return sqrt(x)


cdef inline DTYPE_t reduced_from_euclidean(DTYPE_t x, dist_params* params):
    return x * x


cdef inline DTYPE_t minkowski_from_reduced(DTYPE_t x, dist_params* params):
    return pow(x, 1. / params.minkowski.p)


cdef inline DTYPE_t reduced_from_minkowski(DTYPE_t x, dist_params* params):
    return pow(x, params.minkowski.p)


cdef inline dist_func get_reduced_dfunc(dist_func dfunc):
    if dfunc == &euclidean_distance:
        return &sqeuclidean_distance
    elif dfunc == &seuclidean_distance:
        return &sqseuclidean_distance
    elif dfunc == &minkowski_distance:
        return &pminkowski_distance
    elif dfunc == &wminkowski_distance:
        return &pwminkowski_distance
    elif dfunc == &mahalanobis_distance:
        return &sqmahalanobis_distance
    else:
        return dfunc


cdef inline dist_conv_func get_dist_to_reduced(dist_func dfunc):
    if dfunc == &euclidean_distance:
        return &reduced_from_euclidean
    elif dfunc == &seuclidean_distance:
        return &reduced_from_euclidean
    elif dfunc == &minkowski_distance:
        return &reduced_from_minkowski
    elif dfunc == &wminkowski_distance:
        return &reduced_from_minkowski
    elif dfunc == &mahalanobis_distance:
        return &reduced_from_euclidean
    else:
        return &no_conversion


cdef inline dist_conv_func get_reduced_to_dist(dist_func dfunc):
    if dfunc == &euclidean_distance:
        return &euclidean_from_reduced
    elif dfunc == &seuclidean_distance:
        return &euclidean_from_reduced
    elif dfunc == &minkowski_distance:
        return &minkowski_from_reduced
    elif dfunc == &wminkowski_distance:
        return &minkowski_from_reduced
    elif dfunc == &mahalanobis_distance:
        return &euclidean_from_reduced
    else:
        return &no_conversion


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)


###############################################################################
# DistanceMetric class

cdef class DistanceMetric(object):
    def __cinit__(self):
        """Initialize all arrays to empty"""
        self.mahalanobis_VI = self.seuclidean_V = self.minkowski_w =\
            self.norms1 = self.norms2 = self.precentered_data1 =\
            self.precentered_data2 = self.work_buffer = np.ndarray(0)
        self.dfunc = &euclidean_distance

    def __init__(self, metric="euclidean", w=None, p=None, V=None, VI=None):
        """Object for computing distance between points.

        This is a specialized object for efficient distance computations.
        The objects contain C-pointers to fast implementations of the
        distance functions, to support their use in BallTree.

        Parameters
        ----------
        metric : string
            Distance metric (see Notes below)
        w : ndarray
            The weight vector (for weighted Minkowski)
        p : double
            The p-norm to apply (for Minkowski, weighted and unweighted)
        V : ndarray
            The variance vector (for standardized Euclidean)
        VI : ndarray
            The inverse of the covariance matrix (for Mahalanobis)
        
        Notes
        -----
        ``metric`` can be one of the following:

        - Metrics designed for floating-point input:

          - 'euclidean' / 'l2'
          - 'seuclidean'
          - 'manhattan' / 'cityblock' / 'l1'
          - 'chebyshev'
          - 'minkowski'
          - 'wminkowski'
          - 'mahalanobis'
          - 'cosine'
          - 'correlation'
          - 'hamming'
          - 'jaccard'
          - 'canberra'
          - 'braycurtis'

        - non-metrics which can be used for fast distance comparison

          - 'sqeuclidean'
          - 'sqseuclidean'
          - 'pminkowski'
          - 'pwminkowski'
          - 'sqmahalanobis'

        - Metrics designed for boolean input:
        
          - 'yule'
          - 'matching'
          - 'dice'
          - 'kulsinski'
          - 'rogerstanimoto'
          - 'russellrao'
          - 'sokalmichener'
          - 'sokalsneath'
          
        For details on the form of the metrics, see the docstring of
        :class:`distance_metrics`.

        """
        self.metric = metric
        self.init_kwargs = dict(V=V, VI=VI, p=p, w=w)
        self._init_metric(metric, **self.init_kwargs)

    def _init_metric(self, metric, V, VI, p, w):
        self.learn_params_from_data = False

        if metric in ["euclidean", 'l2', None]:
            self.dfunc = &euclidean_distance

        elif metric in ("manhattan", "cityblock", "l1"):
            self.dfunc = &manhattan_distance

        elif metric == "chebyshev":
            self.dfunc = &chebyshev_distance

        elif metric == "minkowski":
            if p == None:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter p must be specified.")
            elif p <= 0:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter p must be greater than 0.")
            elif p == 1:
                self.dfunc = &manhattan_distance

            elif p == 2:
                self.dfunc = &euclidean_distance

            elif p == np.inf:
                self.dfunc = &chebyshev_distance

            else:
                self.dfunc = &minkowski_distance
                self.params.minkowski.p = p

        elif metric == "pminkowski":
            if p == None:
                raise ValueError("For metric = 'pminkowski', "
                                 "parameter p must be specified.")
            elif p <= 0:
                raise ValueError("For metric = 'pminkowski', "
                                 "parameter p must be greater than 0.")
            elif p == 1:
                self.dfunc = &manhattan_distance

            elif p == 2:
                self.dfunc = &sqeuclidean_distance

            elif p == np.inf:
                self.dfunc = &chebyshev_distance

            else:
                self.dfunc = &pminkowski_distance
                self.params.minkowski.p = p

        elif metric == "wminkowski":
            self.dfunc = &wminkowski_distance

            if p == None:
                raise ValueError("For metric = 'wminkowski', "
                                 "parameter p must be specified.")
            elif p <= 0:
                raise ValueError("For metric = 'wminkowski', "
                                 "parameter p must be greater than 0.")
            self.params.minkowski.p = p

            if w is None:
                raise ValueError("For metric = 'wminkowski', "
                                 "parameter w must be specified.")
            self.minkowski_w = np.asarray(w, dtype=DTYPE, order='C')
            assert self.minkowski_w.ndim == 1
            self.params.minkowski.w = <DTYPE_t*>self.minkowski_w.data
            self.params.minkowski.n = self.minkowski_w.shape[0]

        elif metric == "pwminkowski":
            self.dfunc = &pwminkowski_distance

            if p == None:
                raise ValueError("For metric = 'pwminkowski', "
                                 "parameter p must be specified.")
            elif p <= 0:
                raise ValueError("For metric = 'pwminkowski', "
                                 "parameter p must be greater than 0.")
            self.params.minkowski.p = p

            if w is None:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter w must be specified.")
            self.minkowski_w = np.asarray(w, dtype=DTYPE, order='C')
            assert self.minkowski_w.ndim == 1
            self.params.minkowski.w = <DTYPE_t*>self.minkowski_w.data
            self.params.minkowski.n = self.minkowski_w.shape[0]

        elif metric in ["sqmahalanobis", "mahalanobis"]:
            if VI is None:
                self.learn_params_from_data = True
            else:
                self.mahalanobis_VI = np.asarray(VI, dtype=DTYPE, order='C')
                assert self.mahalanobis_VI.ndim == 2
                assert (self.mahalanobis_VI.shape[0]
                        == self.mahalanobis_VI.shape[1])
                self.work_buffer = np.empty(
                    self.mahalanobis_VI.shape[0], dtype=DTYPE)

                self.params.mahalanobis.n = self.mahalanobis_VI.shape[0]
                self.params.mahalanobis.VI = \
                    <DTYPE_t*> self.mahalanobis_VI.data
                self.params.mahalanobis.work_buffer = \
                    <DTYPE_t*> self.work_buffer.data

            if metric == "mahalanobis":
                self.dfunc = &mahalanobis_distance
            else:
                self.dfunc = &sqmahalanobis_distance

        elif metric in ['seuclidean', 'sqseuclidean']:
            if V is None:
                self.learn_params_from_data = True
            else:
                self.seuclidean_V = np.asarray(V)
                assert self.seuclidean_V.ndim == 1
                
                self.params.seuclidean.V = <DTYPE_t*> self.seuclidean_V.data
                self.params.seuclidean.n = self.seuclidean_V.shape[0]
            if metric == 'seuclidean':
                self.dfunc = &seuclidean_distance
            else:
                self.dfunc = &sqseuclidean_distance

        elif metric == 'sqeuclidean':
            self.dfunc = &sqeuclidean_distance

        elif metric == 'cosine':
            self.params.cosine.precomputed_norms = 0
            self.dfunc = &cosine_distance

        elif metric == 'correlation':
            self.params.correlation.precomputed_data = 0
            self.dfunc = &correlation_distance

        elif metric == 'hamming':
            self.dfunc = &hamming_distance

        elif metric == 'jaccard':
            self.dfunc = &jaccard_distance

        elif metric == 'canberra':
            self.dfunc = &canberra_distance

        elif metric == 'braycurtis':
            self.dfunc = &braycurtis_distance

        elif metric == 'yule':
            self.dfunc = &yule_distance

        elif metric == 'matching':
            self.dfunc = &matching_distance

        elif metric == 'dice':
            self.dfunc = dice_distance

        elif metric == 'kulsinski':
            self.dfunc = &kulsinski_distance

        elif metric == 'rogerstanimoto':
            self.dfunc = &rogerstanimoto_distance

        elif metric == 'russellrao':
            self.dfunc = &russellrao_distance

        elif metric == 'sokalmichener':
            self.dfunc = &sokalmichener_distance

        elif metric == 'sokalsneath':
            self.dfunc = &sokalsneath_distance

        elif callable(metric):
            x = np.random.random(3)
            try:
                res = float(metric(x, x))
            except:
                raise ValueError("user-defined metrics must accept two "
                                 "vectors and return a scalar.")
            self.params.user.func = <void*> metric
            self.dfunc = &user_distance

        else:
            raise ValueError('unrecognized metric %s' % metric)

        self.reduced_dfunc = get_reduced_dfunc(self.dfunc)
        self.dist_to_reduced = get_dist_to_reduced(self.dfunc)
        self.reduced_to_dist = get_reduced_to_dist(self.dfunc)

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (DistanceMetric,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        return (self.metric, self.init_kwargs)

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.metric = state[0]
        self.init_kwargs = state[1]
        self._init_metric(state[0], **(state[1]))

    def set_params_from_data(self, X1, X2 = None, persist=True):
        """Set internal parameters from data

        Some distance metrics require extra information, which can be
        learned from the data matrices.  This function sets those
        internal parameters

        Parameters
        ----------
        X1 : array-like
        X2 : array-like (optional, default = None)
        persist : bool (optional, default = True)
            if False, the parameters will be recomputed on the new data
            each time another distance measurement is performed.
            if True, the parameters will persist for all future distance
            computations
        """
        if persist:
            self.learn_params_from_data = False

        X1 = np.asarray(X1)
        if X2 is not None:
            X2 = np.asarray(X2)

        if ((self.dfunc == &mahalanobis_distance)
            or (self.dfunc == &sqmahalanobis_distance)):
            # compute covariance matrix from data
            if X2 is None:
                X = X1
            else:
                X = np.vstack((X1, X2))

            V = np.cov(X.T)

            if X.shape[0] < X.shape[1]:
                warnings.warn('Mahalanobis Distance: singular covariance '
                              'matrix.  Using pseudo-inverse')
                self.mahalanobis_VI = np.linalg.pinv(V).T
            else:
                self.mahalanobis_VI = np.linalg.inv(V).T

            self.work_buffer = np.zeros(V.shape[0])
            self.params.mahalanobis.n = V.shape[0]
            self.params.mahalanobis.VI = \
                <DTYPE_t*>self.mahalanobis_VI.data
            self.params.mahalanobis.work_buffer = \
                <DTYPE_t*>self.work_buffer.data

        elif ((self.dfunc == &seuclidean_distance)
              or (self.dfunc == &sqseuclidean_distance)):
            # compute variance from data
            if X2 is None:
                X = X1
            else:
                X = np.vstack((X1, X2))

            self.seuclidean_V = X.var(axis=0, ddof=1)
            self.params.seuclidean.V = <DTYPE_t*> self.seuclidean_V.data
            self.params.seuclidean.n = self.seuclidean_V.shape[0]

    def precompute_params_from_data(self, X1, X2=None):
        """Precompute parameters for faster distance computation

        Parameters
        ----------
        X1 : array-like
        X2 : array-like (optional, default = None)
        """
        X1 = np.asarray(X1, dtype=DTYPE, order='C')
        if X2 is not None:
            X2 = np.asarray(X2, dtype=DTYPE, order='C')

        if self.dfunc == &cosine_distance:
            self.params.cosine.precomputed_norms = 1
            self.norms1 = _norms(X1)
            self.params.cosine.norms1 = <DTYPE_t*> self.norms1.data
            if X2 is None:
                self.params.cosine.norms2 = self.params.cosine.norms1
            else:
                self.norms2 = _norms(X2)
                self.params.cosine.norms2 = <DTYPE_t*> self.norms2.data

        elif self.dfunc == &correlation_distance:
            self.params.correlation.precomputed_data = 1
            self.precentered_data1 = _centered(X1)
            self.norms1 = _norms(self.precentered_data1)
            self.params.correlation.x1 = \
                <DTYPE_t*> self.precentered_data1.data
            self.params.correlation.norms1 = <DTYPE_t*> self.norms1.data
            if X2 is None:
                self.params.correlation.x2 = self.params.correlation.x1
                self.params.correlation.norms2 = self.params.correlation.norms1
            else:
                self.precentered_data2 = _centered(X2)
                self.norms2 = _norms(self.precentered_data2)
                self.params.correlation.x2 = \
                    <DTYPE_t*> self.precentered_data2.data
                self.params.correlation.norms2 = <DTYPE_t*> self.norms2.data


    def _check_input(self, X1, X2=None, squareform=False):
        """Internal function to check inputs and convert to appropriate form.

        In addition, for some metrics this function verifies special
        requirements deriving from initialization.

        Parameters
        ----------
        X1 : array-like
        X2 : array-like (optional, default = None)
        squareform : bool
            specify whether Y is square form.  Used only if X2 is None
 
        
        Returns
        -------
        (X1, Y) if X2 is None
        (X1, X2, Y) if X2 is not None

        Raises
        ------
        ValueError
            if inputs cannot be converted to the correct form
        """
        X1 = np.asarray(X1, dtype=DTYPE, order='C')
        assert X1.ndim == 2
        m1 = m2 = X1.shape[0]
        n = X1.shape[1]

        if X2 is not None:
            X2 = np.asarray(X2, dtype=DTYPE, order='C')
            assert X2.ndim == 2
            assert X2.shape[1] == n
            m2 = X2.shape[0]

        if self.learn_params_from_data:
            self.set_params_from_data(X1, X2, persist=False)

        self.precompute_params_from_data(X1, X2)

        # check that data matches metric
        if ((self.dfunc == &mahalanobis_distance)
            or (self.dfunc == &sqmahalanobis_distance)):
            assert n == self.params.mahalanobis.n

        elif ((self.dfunc == &seuclidean_distance)
              or (self.dfunc == &sqseuclidean_distance)):
            assert n == self.params.seuclidean.n

        elif self.dfunc == &wminkowski_distance:
            assert n == self.params.minkowski.n

        elif self.dfunc == &pwminkowski_distance:
            assert n == self.params.minkowski.n

        elif self.dfunc == &cosine_distance:
            self.params.cosine.precomputed_norms = 1
            self.norms1 = _norms(X1)
            self.params.cosine.norms1 = <DTYPE_t*> self.norms1.data
            if X2 is None:
                self.params.cosine.norms2 = self.params.cosine.norms1
            else:
                self.norms2 = _norms(X2)
                self.params.cosine.norms2 = <DTYPE_t*> self.norms2.data

        # construct matrix Y
        if X2 is None:
            if squareform:
                Y = np.empty((m1, m2), dtype=DTYPE)
            else:
                Y = np.empty(m1 * (m1 - 1) / 2, dtype=DTYPE)
            return X1, Y
        else:
            Y = np.empty((m1, m2), dtype=DTYPE)
            return X1, X2, Y

    def _cleanup_after_computation(self):
        if self.dfunc == &cosine_distance:
            self.params.cosine.precomputed_norms = 0
        elif self.dfunc == &correlation_distance:
            self.params.correlation.precomputed_data = 0
            self.precentered_data1 = np.ndarray(0)
            self.precentered_data2 = np.ndarray(0)

    def cdist(self, X1, X2):
        """Compute the distance between each pair of observation vectors.

        Parameters
        ----------
        X1 : array-like, shape = (m1, n)
        X2 : array-like, shape = (m2, n)

        Returns
        -------
        Y : ndarray, shape = (m1, m2)
            Y[i, j] is the distance between the vectors X1[i] and X2[j]
            evaluated with the appropriate distance metric.

        See Also
        --------
        scipy.spatial.distance.cdist
        """
        X1, X2, Y = self._check_input(X1, X2)

        self._cdist_cc(X1, X2, Y)

        self._cleanup_after_computation()

        return Y
                        
    def pdist(self, X, squareform=False):
        """Compute the pairwise distances between each pair of vectors.

        Parameters
        ----------
        X : array-like, shape = (m, n)

        squareform : boolean (optional)
            if true, return the squareform of the matrix
        
        Returns
        -------
        Y : ndarray
            Array of distances
            
            - if squareform == False (default value), then
              Y is a 1-D array of size (m * (m - 1) / 2),
              a compact representation of the distance matrix.
              It can be converted to squareform using the function
              ``scipy.spatial.distance.squareform``

            - if squareform == True, then Y is of shape (m, m) and
              Y[i, j] is the distance between the vectors X[i] and X[j]
              evaluated with the appropriate distance metric.

        See Also
        --------
        scipy.spatial.distance.pdist
        """
        X, Y = self._check_input(X, squareform=squareform)

        if squareform:
            self._pdist_c(X, Y)
        else:
            self._pdist_c_compact(X, Y)

        self._cleanup_after_computation()

        return Y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _cdist_cc(self,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] X1,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] X2,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] Y):
        # cdist() workhorse.  '_cc' means X1 & X2 are c-ordered
        # arrays are assumed to have:
        #   X1.shape == (m1, n)
        #   X2.shape == (m2, n)
        #   Y.shape == (m1, m2)
        # this is not checked within the function.
        cdef Py_ssize_t i1, i2, m1, m2, n
        m1 = X1.shape[0]
        m2 = X2.shape[0]
        n = X1.shape[1]

        cdef DTYPE_t* pX1 = <DTYPE_t*> X1.data
        cdef DTYPE_t* pX2 = <DTYPE_t*> X2.data

        for i1 from 0 <= i1 < m1:
            pX2 = <DTYPE_t*> X2.data
            for i2 from 0 <= i2 < m2:
                Y[i1, i2] = self.dfunc(pX1,
                                       pX2, n,
                                       &self.params, i1, i2)
                pX2 += n
            pX1 += n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _pdist_c(self,
                       np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                       np.ndarray[DTYPE_t, ndim=2, mode='c'] Y):
        # pdist() workhorse.  '_c' means X is c-ordered
        # arrays are assumed to have:
        #   X.shape == (m, n)
        #   Y.shape == (m, m)
        # this is not checked within the function.
        cdef Py_ssize_t i1, i2, m, n
        m = Y.shape[0]
        n = X.shape[1]

        cdef DTYPE_t* pX1 = <DTYPE_t*> X.data
        cdef DTYPE_t* pX2 = <DTYPE_t*> X.data

        for i1 from 0 <= i1 < m:
            Y[i1, i1] = 0
            pX2 = pX1 + n
            for i2 from i1 < i2 < m:
                Y[i1, i2] = Y[i2, i1] = self.dfunc(pX1,
                                                   pX2, n,
                                                   &self.params, i1, i2)
                pX2 += n
            pX1 += n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _pdist_c_compact(self,
                               np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] Y):
        # pdist() workhorse.  '_c' means X is c-ordered
        #                     '_compact' means Y is in compact form
        # arrays are assumed to have:
        #   X.shape == (m, n)
        #   Y.size == m * (m - 1) / 2
        # this is not checked within the function.
        cdef Py_ssize_t i, i1, i2, m, n
        m = X.shape[0]
        n = X.shape[1]

        cdef DTYPE_t* pX1 = <DTYPE_t*> X.data
        cdef DTYPE_t* pX2 = <DTYPE_t*> X.data
        i = 0
        for i1 from 0 <= i1 < m:
            pX2 = pX1 + n
            for i2 from i1 < i2 < m:
                Y[i] = self.dfunc(pX1, pX2, n,
                                  &self.params, i1, i2)
                pX2 += n
                i += 1
            pX1 += n


def pairwise_distances(X1, X2=None, metric="euclidean", **kwargs):
    """Compute pairwise distances in the given metric

    Parameters
    ----------
    X1, X2 : array-like
        arrays of row vectors.  If X2 is not specified, use X1 = X2
    metric : string
        distance metric
    
    Other Parameters
    ----------------
    V, VI, w, p : specific to metric

    Returns
    -------
    D : ndarray or float
        distances between the points in x1 and x2
        shape is X1.shape[:-1] + X2.shape[:-1]
    
    """
    dist_metric = DistanceMetric(metric, **kwargs)

    X1 = np.asarray(X1, dtype=DTYPE, order='C')
    shape1 = X1.shape
    X1 = X1.reshape((-1, shape1[-1]))

    if X2 is None or X1 is X2:
        shape2 = shape1
        Y = dist_metric.pdist(X1)

    else:
        X2 = np.asarray(X2, dtype=DTYPE, order='C')
        shape2 = X2.shape
        assert shape1[-1] == shape2[-1]
        X2 = X2.reshape((-1, shape2[-1]))

        Y = dist_metric.cdist(X1, X2)

    return Y.reshape(shape1[:-1] + shape2[:-1])


# TODO: finish these definitions and document them
euclidean = lambda x1, x2: \
    pairwise_distances(x1, x2, "euclidean")
l2 = lambda x1, x2: \
    pairwise_distances(x1, x2, "l2")
manhattan = lambda x1, x2: \
    pairwise_distances(x1, x2, "manhattan")
cityblock = lambda x1, x2: \
    pairwise_distances(x1, x2, "cityblock")
l1 = lambda x1, x2: \
    pairwise_distances(x1, x2, "l1")
chebyshev = lambda x1, x2: \
    pairwise_distances(x1, x2, "chebyshev")
minkowski = lambda x1, x2, p=None: \
    pairwise_distances(x1, x2, "minkowski", p=p)
wminkowski = lambda x1, x2, p=None, w=None: \
    pairwise_distances(x1, x2, "wminkowski", p=p, w=w)

