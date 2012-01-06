import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool

from libc.math cimport fabs, fmax

# TODO:
#  Functionality:
#   - implement within BallTree
#   - allow compact form for pdist (similar to scipy.spatial.distance.pdist)
#   - allow user-defined functions as metrics (is this possible?)
#
#  Speed:
#   - use blas for computations where appropriate
#   - speed up multiple computations with cosine distance & correlation
#     distance.  Currently these will repeatedly compute the norm/mean of
#     the vectors.
#
#  Memory:
#   - make cdist/pdist work with fortran arrays
#   - make cdist/pdist work with csr matrices.  This will require writing
#     a new form of each distance function which accepts csr input.
#
#  Documentation:
#   - documentation of metrics
#   - double-check comparison with sklearn.metrics
#
#  Templating?
#   - this would be a great candidate to try out cython templating

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

###############################################################################
# Define data structures needed for distance calculations

# data structure used for mahalanobis distance
cdef struct mahalanobis_info:
    ITYPE_t n             # size of arrays
    DTYPE_t* VI   # pointer to buffer of size n * n
    DTYPE_t* work_buffer  # pointer to buffer of size n

# data structure used for (weighted) minkowski distance
cdef struct minkowski_info:
    ITYPE_t n   # size of array
    DTYPE_t p   # specifies p-norm
    DTYPE_t* w  # pointer to buffer of size n

# data structure used for standardized euclidean distance
cdef struct seuclidean_info:
    ITYPE_t n   # size of array
    DTYPE_t* V  # pointer to buffer of size n

# general distance data structure.  We use a union because
# different distance metrics require different ancillary information.
cdef union dist_params:
    minkowski_info minkowski
    mahalanobis_info mahalanobis
    seuclidean_info seuclidean

# define a pointer to a general distance function.
ctypedef DTYPE_t (*dist_func)(DTYPE_t*, DTYPE_t*, ITYPE_t, dist_params*)


###############################################################################
# Define the various distance functions

cdef DTYPE_t euclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t d, res = 0
    
    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d
    
    return res ** 0.5


cdef DTYPE_t manhattan_distance(DTYPE_t* x1, DTYPE_t* x2,
                                ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t res = 0
    
    for i from 0 <= i < n:
        res += fabs(x1[i] - x2[i])

    return res


cdef DTYPE_t chebyshev_distance(DTYPE_t* x1, DTYPE_t* x2,
                                ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t res = 0
    
    for i from 0 <= i < n:
        res = fmax(res, fabs(x1[i] - x2[i]))

    return res


cdef DTYPE_t minkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t d, res = 0

    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += d ** params.minkowski.p

    return res ** (1. / params.minkowski.p)


cdef DTYPE_t wminkowski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t d, res = 0
    
    for i from 0 <= i < n:
        d = fabs(x1[i] - x2[i])
        res += (params.minkowski.w[i] * d) ** params.minkowski.p

    return res ** (1. / params.minkowski.p)


cdef DTYPE_t mahalanobis_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  ITYPE_t n, dist_params* params):
    cdef ITYPE_t i, j
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

    return res ** 0.5


cdef DTYPE_t seuclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t d, res = 0
    
    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d / params.seuclidean.V[i]
    
    return res ** 0.5


cdef DTYPE_t sqeuclidean_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t d, res = 0

    for i from 0 <= i < n:
        d = x1[i] - x2[i]
        res += d * d
    
    return res


cdef DTYPE_t cosine_distance(DTYPE_t* x1, DTYPE_t* x2,
                             ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t x1nrm = 0, x2nrm = 0, x1Tx2 = 0

    # TODO : think about how to speed this up for cdist & pdist by
    #        only computing the norm once per point

    # TODO: use blas here
    for i from 0 <= i < n:
        x1nrm += x1[i] * x1[i]
        x2nrm += x2[i] * x2[i]
        x1Tx2 += x1[i] * x2[i]

    return 1.0 - (x1Tx2) / np.sqrt(x1nrm * x2nrm)


cdef DTYPE_t correlation_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t mu1 = 0, mu2 = 0, x1nrm = 0, x2nrm = 0, x1Tx2 = 0

    cdef DTYPE_t tmp1, tmp2

    # TODO : think about how to speed this up for cdist & pdist by
    #        only computing the mean once per point
    for i from 0 <= i < n:
        mu1 += x1[i]
        mu2 += x2[i]
    mu1 /= n
    mu2 /= n

    # TODO : use blas here
    for i from 0 <= i < n:
        tmp1 = x1[i] - mu1
        tmp2 = x2[i] - mu2
        x1nrm += tmp1 * tmp1
        x2nrm += tmp2 * tmp2
        x1Tx2 += tmp1 * tmp2

    return 1. - x1Tx2 / np.sqrt(x1nrm * x2nrm)


cdef DTYPE_t hamming_distance(DTYPE_t* x1, DTYPE_t* x2,
                              ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef ITYPE_t n_disagree = 0

    for i from 0 <= i < n:
        n_disagree += (x1[i] != x2[i])

    return n_disagree * 1. / n


cdef DTYPE_t jaccard_distance(DTYPE_t* x1, DTYPE_t* x2,
                              ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef ITYPE_t n_disagree = 0

    for i from 0 <= i < n:
        if x1[i] != 0:
            if x2[i] != 0:
                if (x1[i] != x2[i]):
                    n_disagree += 1

    return n_disagree * 1. / n


cdef DTYPE_t canberra_distance(DTYPE_t* x1, DTYPE_t* x2,
                               ITYPE_t n, dist_params* params):
    cdef DTYPE_t res = 0, denominator
    cdef ITYPE_t i

    for i from 0 <= i < n:
        denominator = (fabs(x1[i]) + fabs(x2[i]))
        if denominator > 0:
            res += fabs(x1[i] - x2[i]) / denominator

    return res


cdef DTYPE_t braycurtis_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 ITYPE_t n, dist_params* params):
    cdef ITYPE_t i
    cdef DTYPE_t numerator = 0, denominator = 0

    for i from 0 <= i < n:
        numerator += fabs(x1[i] - x2[i])
        denominator += fabs(x1[i])
        denominator += fabs(x2[i])

    return numerator / denominator


cdef DTYPE_t yule_distance(DTYPE_t* x1, DTYPE_t* x2,
                           ITYPE_t n, dist_params* params):
    cdef ITYPE_t ntt = 0, nff = 0, ntf = 0, nft = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        nff += (1 - TF1) * (1 - TF2)
        nft += (1 - TF1) * TF2
        ntf += TF1 * (1 - TF2)
        ntt += TF1 * TF2

    return (2. * ntf * nft) / (1. * (ntt * nff + ntf * nft))


cdef DTYPE_t matching_distance(DTYPE_t* x1, DTYPE_t* x2,
                               ITYPE_t n, dist_params* params):
    cdef ITYPE_t n_neq = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        n_neq += (TF1 != TF2)

    return n_neq * 1. / n


cdef DTYPE_t dice_distance(DTYPE_t* x1, DTYPE_t* x2,
                           ITYPE_t n, dist_params* params):
    cdef ITYPE_t ntt = 0, n_neq = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2
        n_neq += (TF1 != TF2)

    return n_neq * 1. / (2 * ntt + n_neq)


cdef DTYPE_t kulsinski_distance(DTYPE_t* x1, DTYPE_t* x2,
                                ITYPE_t n, dist_params* params):
    cdef ITYPE_t ntt = 0, n_neq = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2
        n_neq += (TF1 != TF2)

    return (n_neq - ntt + n) * 1. / (n_neq + n)


cdef DTYPE_t rogerstanimoto_distance(DTYPE_t* x1, DTYPE_t* x2,
                                     ITYPE_t n, dist_params* params):
    cdef ITYPE_t n_neq = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        n_neq += (TF1 != TF2)

    return n_neq * 2. / (n + n_neq)


cdef DTYPE_t russellrao_distance(DTYPE_t* x1, DTYPE_t* x2,
                                 ITYPE_t n, dist_params* params):
    cdef ITYPE_t ntt = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2

    return (n - ntt) * 1. / n


cdef DTYPE_t sokalmichener_distance(DTYPE_t* x1, DTYPE_t* x2,
                                    ITYPE_t n, dist_params* params):
    cdef ITYPE_t n_neq = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        n_neq += (TF1 != TF2)

    return n_neq * 2.0 / (n + n_neq)


cdef DTYPE_t sokalsneath_distance(DTYPE_t* x1, DTYPE_t* x2,
                                  ITYPE_t n, dist_params* params):
    cdef ITYPE_t ntt = 0, n_neq = 0
    cdef ITYPE_t i
    cdef ITYPE_t TF1, TF2

    for i from 0 <= i < n:
        TF1 = (x1[i] != 0)
        TF2 = (x2[i] != 0)
        ntt += TF1 * TF2
        n_neq += (TF1 != TF2)

    return n_neq * 2.0 / (ntt + 2 * n_neq)
           

cdef class DistanceMetric(object):
    # C attributes which store information about the distance function
    cdef dist_params params
    cdef dist_func dfunc

    # array attributes used for various distance measures
    cdef np.ndarray mahalanobis_VI # inverse covariance matrix of data
    cdef np.ndarray seuclidean_V   # variance array of data
    cdef np.ndarray minkowski_w    # weights for weighted minkowski
    cdef np.ndarray work_buffer    # work buffer

    # flags for computation
    cdef int learn_params_from_data

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
            self.dfunc = &cosine_distance

        elif metric == 'correlation':
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

        Y = np.empty((m1, m2), dtype=DTYPE)

        if X2 is None:
            return X1, Y
        else:
            return X1, X2, Y

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
                Y[i1, i2] = self.dfunc(pX1 + i1 * n,
                                       pX2 + i2 * n,
                                       n, &self.params)

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
                Y[i1, i2] = Y[i2, i1] = self.dfunc(pX + i1 * n,
                                                   pX + i2 * n,
                                                   n, &self.params)


def distance(x1, x2, metric="euclidean", **kwargs):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    shape1 = x1.shape
    shape2 = x2.shape

    x1 = x1.reshape((-1, shape1[-1]))
    x2 = x2.reshape((-1, shape2[-1]))

    dist_metric = DistanceMetric(metric, **kwargs)
    Y = dist_metric.cdist(x1, x2)

    return Y.reshape(shape1[:-1] + shape2[:-1])
