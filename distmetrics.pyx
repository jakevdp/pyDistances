import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fabs, fmax, sqrt, pow

# python data types (corresponding C-types are in pxd file)
DTYPE = np.float64

# TODO:
#  Functionality:
#   - implement within BallTree
#   - allow compact output from pdist (similar to scipy.spatial.distance.pdist)
#
#  Speed:
#   - use blas for computations where appropriate
#   - boolean functions are slow: how do we access fast C boolean operations?
#
#  Memory & storage:
#   - make cdist/pdist work with fortran arrays (see note below)
#   - make cdist/pdist work with csr matrices.  This will require writing
#     a new form of each distance function which accepts csr input.
#   - figure out how to wrap a memory array with a temporary numpy array
#     (buffer_to_array function, below)
#
#  Documentation:
#   - documentation of metrics
#   - double-check consistency with sklearn.metrics & scipy.spatial.distance
#
#  Templating?
#   - this would be a great candidate to try out cython templating
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

# TODO: figure out how to do this without copying data
cdef np.ndarray _buffer_to_ndarray(DTYPE_t* x, Py_ssize_t n):
    cdef np.ndarray y = np.empty(n, dtype=DTYPE)
    cdef DTYPE_t* ydata = <DTYPE_t*> y.data
    cdef Py_ssize_t i
    for i from 0 <= i < n:
        ydata[i] = x[i]
    return y


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
#         pointers to data arrays (see notes below)
#     n : integer
#         length of vector (see notes below)
#     params : structure
#         the parameter structure contains various parameters that define
#         the distance metric, or aid in faster computation.
#     rowindex1, rowindex2 : integers
#         these define the offsets where the data starts (see notes below)
#    
#     Returns
#     -------
#     D : double
#         distance between v1 and v2
#    
#     Notes
#     -----
#     the data in the vectors v1 and v2 are defined by the following 
#     locations in memory:
#
#     - v1 = x1[n * row_offset1 : (n + 1) * rowindex1]
#     - v2 = x2[n * row_offset2 : (n + 1) * rowindex2]
#
#     passing rowindex1 and rowindex2 becomes useful for metrics where
#     computation can be made more efficient by precomputing information
#     about each point: e.g. the mean and norm in cosine_distance and
#     correlation_distance, etc.
###############################################################################

cdef DTYPE_t euclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n
    
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

    x1 += rowindex1 * n
    x2 += rowindex2 * n
    
    for i from 0 <= i < n:
        res += fabs(x1[i] - x2[i])

    return res


cdef DTYPE_t chebyshev_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t res = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n
    
    for i from 0 <= i < n:
        res = fmax(res, fabs(x1[i] - x2[i]))

    return res


cdef DTYPE_t minkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                Py_ssize_t n, dist_params* params,
                                Py_ssize_t rowindex1,
                                Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n

    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += pow(d, params.minkowski.p)

    return pow(res, 1. / params.minkowski.p)


cdef DTYPE_t wminkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n
    
    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += pow(params.minkowski.w[i] * d, params.minkowski.p)

    return pow(res, 1. / params.minkowski.p)


cdef DTYPE_t mahalanobis_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  Py_ssize_t n, dist_params* params,
                                  Py_ssize_t rowindex1,
                                  Py_ssize_t rowindex2):
    cdef Py_ssize_t i, j
    cdef DTYPE_t d, res = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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


cdef DTYPE_t seuclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 Py_ssize_t n, dist_params* params,
                                 Py_ssize_t rowindex1,
                                 Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n
    
    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d / params.seuclidean.V[i]
    
    return sqrt(res)


cdef DTYPE_t sqeuclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  Py_ssize_t n, dist_params* params,
                                  Py_ssize_t rowindex1,
                                  Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef DTYPE_t d, res = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

    # TODO: use blas here

    if params.cosine.precomputed_norms:
        for i from 0 <= i < n:
            x1Tx2 += x1[i] * x2[i]
        x1nrm = params.cosine.norms1[rowindex1]
        x2nrm = params.cosine.norms2[rowindex2]
        normalization = x1nrm * x2nrm
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

    # TODO : use blas here
    if params.correlation.precomputed_data:
        x1 = params.correlation.x1 + rowindex1 * n
        x2 = params.correlation.x2 + rowindex2 * n
        for i from 0 <= i < n:
            x1Tx2 += x1[i] * x2[i]
        
        normalization = params.correlation.norms1[rowindex1] 
        normalization *= params.correlation.norms2[rowindex2]
    else:
        x1 += rowindex1 * n
        x2 += rowindex2 * n
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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

    for i from 0 <= i < n:
        n_disagree += (x1[i] != x2[i])

    return n_disagree * 1. / n


cdef DTYPE_t jaccard_distance(DTYPE_t* x1, DTYPE_t* x2,
                              Py_ssize_t n, dist_params* params,
                              Py_ssize_t rowindex1,
                              Py_ssize_t rowindex2):
    cdef Py_ssize_t i
    cdef int n_disagree = 0

    x1 += rowindex1 * n
    x2 += rowindex2 * n

    for i from 0 <= i < n:
        if x1[i] != 0:
            if x2[i] != 0:
                if (x1[i] != x2[i]):
                    n_disagree += 1

    return n_disagree * 1. / n


cdef DTYPE_t canberra_distance(DTYPE_t* x1, DTYPE_t* x2,
                               Py_ssize_t n, dist_params* params,
                               Py_ssize_t rowindex1,
                               Py_ssize_t rowindex2):
    cdef DTYPE_t res = 0, denominator
    cdef Py_ssize_t i

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

    for i from 0 <= i < n:
        numerator += fabs(x1[i] - x2[i])
        denominator += fabs(x1[i])
        denominator += fabs(x2[i])

    return numerator / denominator


cdef DTYPE_t yule_distance(DTYPE_t* x1, DTYPE_t* x2,
                           Py_ssize_t n, dist_params* params,
                           Py_ssize_t rowindex1,
                           Py_ssize_t rowindex2):
    cdef int TF1, TF2, ntt = 0, nff = 0, ntf = 0, nft = 0
    cdef Py_ssize_t i

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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

    x1 += rowindex1 * n
    x2 += rowindex2 * n

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
    cdef np.ndarray y1 = _buffer_to_ndarray(x1 + rowindex1 * n, n)
    cdef np.ndarray y2 = _buffer_to_ndarray(x2 + rowindex2 * n, n)
    return (<object>(params.user.func))(y1, y2)
           

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
          - 'manhattan' / 'cityblock' / 'l1'
          - 'chebyshev'
          - 'minkowski'
          - 'wminkowski'
          - 'mahalanobis'
          - 'seuclidean'
          - 'sqeuclidean'
          - 'cosine'
          - 'correlation'
          - 'hamming'
          - 'jaccard'
          - 'canberra'
          - 'braycurtis'

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

        elif metric == "wminkowski":
            self.dfunc = &wminkowski_distance

            if p == None:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter p must be specified.")
            elif p <= 0:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter p must be greater than 0.")
            self.params.minkowski.p = p

            if w is None:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter w must be specified.")
            self.minkowski_w = np.asarray(w, dtype=DTYPE, order='C')
            assert self.minkowski_w.ndim == 1
            self.params.minkowski.w = <DTYPE_t*>self.minkowski_w.data
            self.params.minkowski.n = self.minkowski_w.shape[0]

        elif metric == "mahalanobis":
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
            self.dfunc = &mahalanobis_distance

        elif metric == 'seuclidean':
            if V is None:
                self.learn_params_from_data = True
            else:
                self.seuclidean_V = np.asarray(V)
                assert self.seuclidean_V.ndim == 1
                
                self.params.seuclidean.V = <DTYPE_t*> self.seuclidean_V.data
                self.params.seuclidean.n = self.seuclidean_V.shape[0]
            self.dfunc = &seuclidean_distance

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

    def _check_input(self, X1, X2=None):
        """Internal function to check inputs and convert to appropriate form.

        In addition, for some metrics this function verifies special
        requirements deriving from initialization.

        Parameters
        ----------
        X1 : array-like
        X2 : array-like (optional, default = None)
        
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

        if self.dfunc == &mahalanobis_distance:
            if self.learn_params_from_data:
                # covariance matrix was not specified: compute it from data
                if X2 is None:
                    V = np.cov(X1.T)
                else:
                    V = np.cov(np.vstack((X1, X2)).T)

                # TODO: what about singular matrices?  How to handle this?
                self.mahalanobis_VI = np.linalg.inv(V).T
                self.work_buffer = np.zeros(V.shape[0])
                self.params.mahalanobis.n = n
                self.params.mahalanobis.VI = \
                    <DTYPE_t*>self.mahalanobis_VI.data
                self.params.mahalanobis.work_buffer = \
                    <DTYPE_t*>self.work_buffer.data
                
            assert n == self.params.mahalanobis.n

        elif self.dfunc == &seuclidean_distance:
            if self.learn_params_from_data:
                # variance was not specified: compute it from data
                if X2 is None:
                    self.seuclidean_V = X1.var(axis=0, ddof=1)
                else:
                    self.seuclidean_V = np.vstack((X1, X2)).var(axis=0, ddof=1)
                self.params.seuclidean.V = <DTYPE_t*> self.seuclidean_V.data
                self.params.seuclidean.n = n

            assert n == self.params.seuclidean.n

        elif self.dfunc == &wminkowski_distance:
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

        Y = np.empty((m1, m2), dtype=DTYPE)

        if X2 is None:
            return X1, Y
        else:
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
                        
    def pdist(self, X):
        """Compute the pairwise distances between each pair of vectors.

        Parameters
        ----------
        X : array-like, shape = (m, n)
        
        Returns
        -------
        Y : ndarray, shape = (m, m)
            Y[i, j] is the distance between the vectors X[i] and X[j]
            evaluated with the appropriate distance metric.

        See Also
        --------
        scipy.spatial.distance.pdist
        """
        X, Y = self._check_input(X)

        self._pdist_c(X, Y)

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
        cdef unsigned int i1, i2, m1, m2, n
        m1 = X1.shape[0]
        m2 = X2.shape[0]
        n = X1.shape[1]

        cdef DTYPE_t* pX1 = <DTYPE_t*> X1.data
        cdef DTYPE_t* pX2 = <DTYPE_t*> X2.data

        for i1 from 0 <= i1 < m1:
            for i2 from 0 <= i2 < m2:
                Y[i1, i2] = self.dfunc(pX1, pX2, n,
                                       &self.params, i1, i2)

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
        cdef unsigned int i1, i2, m, n
        m = Y.shape[0]
        n = X.shape[1]

        cdef DTYPE_t* pX = <DTYPE_t*> X.data

        for i1 from 0 <= i1 < m:
            Y[i1, i1] = 0
            for i2 from i1 < i2 < m:
                Y[i1, i2] = Y[i2, i1] = self.dfunc(pX, pX, n,
                                                   &self.params, i1, i2)


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

