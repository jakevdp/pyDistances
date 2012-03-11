import warnings
import numpy as np
from scipy.sparse import isspmatrix, isspmatrix_csr

from utils import safe_asarray, corr, column_var, _norms, _centered

cimport numpy as np
cimport cython

include "distfuncs.pxi"

# python data types (corresponding C-types are in pxd file)
DTYPE = np.float64
ITYPE = np.int32

######################################################################
# TODO:
#  Speed:
#   - use blas for computations where appropriate
#   - boolean functions are slow: how do we access fast C boolean operations?
#     new cython version?
#   - enable fast euclidean distances using (x-y)^2 = x^2 + y^2 - 2xy
#     and 'precomputed_norms' flag
#
#  Documentation:
#   - documentation of metrics
#   - double-check consistency with sklearn.metrics & scipy.spatial.distance
#
#  Future Functionality:
#   - make distances work with fortran arrays (see note below)
#   - make distances work with csr matrices.  This will require writing
#     a new form of each distance function which accepts csr input.
#   - Implement cover tree as well (?)
#   - templating?  this would be a great candidate to try out cython templates.
#
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


@cython.cdivision(True)
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
        self.dfunc_spde = &euclidean_distance_spde
        self.dfunc_spsp = &euclidean_distance_spsp

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

        - User-specified metric: must be a function which accepts two
          numpy ndarrays, and returns a scalar.
          
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
            self.dfunc_spde = &euclidean_distance_spde
            self.dfunc_spsp = &euclidean_distance_spsp

        elif metric in ("manhattan", "cityblock", "l1"):
            self.dfunc = &manhattan_distance
            self.dfunc_spde = &manhattan_distance_spde
            self.dfunc_spsp = &manhattan_distance_spsp

        elif metric == "chebyshev":
            self.dfunc = &chebyshev_distance
            self.dfunc_spde = &chebyshev_distance_spde
            self.dfunc_spsp = &chebyshev_distance_spsp

        elif metric == "minkowski":
            if p == None:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter p must be specified.")
            elif p <= 0:
                raise ValueError("For metric = 'minkowski', "
                                 "parameter p must be greater than 0.")
            elif p == 1:
                self.dfunc = &manhattan_distance
                self.dfunc_spde = &manhattan_distance_spde
                self.dfunc_spsp = &manhattan_distance_spsp

            elif p == 2:
                self.dfunc = &euclidean_distance
                self.dfunc_spde = &euclidean_distance_spde
                self.dfunc_spsp = &euclidean_distance_spsp

            elif p == np.inf:
                self.dfunc = &chebyshev_distance
                self.dfunc_spde = &chebyshev_distance_spde
                self.dfunc_spsp = &chebyshev_distance_spsp

            else:
                self.dfunc = &minkowski_distance
                self.dfunc_spde = &minkowski_distance_spde
                self.dfunc_spsp = &minkowski_distance_spsp
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
                self.dfunc_spde = &manhattan_distance_spde
                self.dfunc_spsp = &manhattan_distance_spsp

            elif p == 2:
                self.dfunc = &sqeuclidean_distance
                self.dfunc_spde = &sqeuclidean_distance_spde
                self.dfunc_spsp = &sqeuclidean_distance_spsp

            elif p == np.inf:
                self.dfunc = &chebyshev_distance
                self.dfunc_spde = &chebyshev_distance_spde
                self.dfunc_spsp = &chebyshev_distance_spsp

            else:
                self.dfunc = &pminkowski_distance
                self.dfunc_spde = &pminkowski_distance_spde
                self.dfunc_spsp = &pminkowski_distance_spsp
                self.params.minkowski.p = p

        elif metric == "wminkowski":
            self.dfunc = &wminkowski_distance
            self.dfunc_spde = &wminkowski_distance_spde
            self.dfunc_spsp = &wminkowski_distance_spsp

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
            self.dfunc_spde = &pwminkowski_distance_spde
            self.dfunc_spsp = &pwminkowski_distance_spsp

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
                self.dfunc_spde = &mahalanobis_distance_spde
                self.dfunc_spsp = &mahalanobis_distance_spsp
            else:
                self.dfunc = &sqmahalanobis_distance
                self.dfunc_spde = &sqmahalanobis_distance_spde
                self.dfunc_spsp = &sqmahalanobis_distance_spsp

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
                self.dfunc_spde = &seuclidean_distance_spde
                self.dfunc_spsp = &seuclidean_distance_spsp
            else:
                self.dfunc = &sqseuclidean_distance
                self.dfunc_spde = &sqseuclidean_distance_spde
                self.dfunc_spsp = &sqseuclidean_distance_spsp

        elif metric == 'sqeuclidean':
            self.dfunc = &sqeuclidean_distance
            self.dfunc_spde = &sqeuclidean_distance_spde
            self.dfunc_spsp = &sqeuclidean_distance_spsp

        elif metric == 'cosine':
            self.params.cosine.precomputed_norms = 0
            self.dfunc = &cosine_distance
            self.dfunc_spde = &cosine_distance_spde
            self.dfunc_spsp = &cosine_distance_spsp

        elif metric == 'correlation':
            self.params.correlation.precomputed_data = 0
            self.dfunc = &correlation_distance
            self.dfunc_spde = &correlation_distance_spde
            self.dfunc_spsp = &correlation_distance_spsp

        elif metric == 'hamming':
            self.dfunc = &hamming_distance
            self.dfunc_spde = &hamming_distance_spde
            self.dfunc_spsp = &hamming_distance_spsp

        elif metric == 'jaccard':
            self.dfunc = &jaccard_distance
            self.dfunc_spde = &jaccard_distance_spde
            self.dfunc_spsp = &jaccard_distance_spsp

        elif metric == 'canberra':
            self.dfunc = &canberra_distance
            self.dfunc_spde = &canberra_distance_spde
            self.dfunc_spsp = &canberra_distance_spsp

        elif metric == 'braycurtis':
            self.dfunc = &braycurtis_distance
            self.dfunc_spde = &braycurtis_distance_spde
            self.dfunc_spsp = &braycurtis_distance_spsp

        elif metric == 'yule':
            self.dfunc = &yule_distance
            self.dfunc_spde = &yule_distance_spde
            self.dfunc_spsp = &yule_distance_spsp

        elif metric == 'matching':
            self.dfunc = &matching_distance
            self.dfunc_spde = &matching_distance_spde
            self.dfunc_spsp = &matching_distance_spsp

        elif metric == 'dice':
            self.dfunc = dice_distance
            self.dfunc_spde = dice_distance_spde
            self.dfunc_spsp = dice_distance_spsp

        elif metric == 'kulsinski':
            self.dfunc = &kulsinski_distance
            self.dfunc_spde = &kulsinski_distance_spde
            self.dfunc_spsp = &kulsinski_distance_spsp

        elif metric == 'rogerstanimoto':
            self.dfunc = &rogerstanimoto_distance
            self.dfunc_spde = &rogerstanimoto_distance_spde
            self.dfunc_spsp = &rogerstanimoto_distance_spsp

        elif metric == 'russellrao':
            self.dfunc = &russellrao_distance
            self.dfunc_spde = &russellrao_distance_spde
            self.dfunc_spsp = &russellrao_distance_spsp

        elif metric == 'sokalmichener':
            self.dfunc = &sokalmichener_distance
            self.dfunc_spde = &sokalmichener_distance_spde
            self.dfunc_spsp = &sokalmichener_distance_spsp

        elif metric == 'sokalsneath':
            self.dfunc = &sokalsneath_distance
            self.dfunc_spde = &sokalsneath_distance_spde
            self.dfunc_spsp = &sokalsneath_distance_spsp

        elif callable(metric):
            x = np.random.random(3)
            try:
                res = float(metric(x, x))
            except:
                raise ValueError("user-defined metrics must accept two "
                                 "vectors and return a scalar.")
            self.params.user.func = <void*> metric
            self.dfunc = &user_distance
            self.dfunc_spde = &user_distance_spde
            self.dfunc_spsp = &user_distance_spsp

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

        X1 = safe_asarray(X1)
        if X2 is not None:
            X2 = safe_asarray(X2)

        if ((self.dfunc == &mahalanobis_distance)
            or (self.dfunc == &sqmahalanobis_distance)):
            # compute covariance matrix from data
            V = corr(X1, X2)
            if V.shape[0] < X1.shape[1]:
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
            self.seuclidean_V = column_var(X1, X2)
            self.params.seuclidean.V = <DTYPE_t*> self.seuclidean_V.data
            self.params.seuclidean.n = self.seuclidean_V.shape[0]

    def precompute_params_from_data(self, X1, X2=None):
        """Precompute parameters for faster distance computation

        Parameters
        ----------
        X1 : array-like
        X2 : array-like (optional, default = None)
        """
        X1 = safe_asarray(X1, dtype=DTYPE, order='C')
        if X2 is not None:
            X2 = safe_asarray(X2, dtype=DTYPE, order='C')

        if isspmatrix(X1) or isspmatrix(X2):
            return

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
        if isspmatrix(X1):
            X1 = X1.tocsr()
        else:
            np.asarray(X1, dtype=DTYPE, order='C')
        assert X1.ndim == 2
        m1 = m2 = X1.shape[0]
        n = X1.shape[1]

        if X2 is not None:
            if isspmatrix(X2):
                X2 = X2.tocsr()
            else:
                np.asarray(X2, dtype=DTYPE, order='C')
            assert X2.ndim == 2
            assert X2.shape[1] == n
            m2 = X2.shape[0]

        if self.learn_params_from_data:
            self.set_params_from_data(X1, X2, persist=False)

        self.precompute_params_from_data(X1, X2)

        # check that metric data matches input
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

        # construct matrix Y
        if X2 is None:
            if squareform:
                Y = np.empty((m1, m1), dtype=DTYPE)
            else:
                Y = np.empty(m1 * (m1 - 1) / 2, dtype=DTYPE)
            return X1, Y
        else:
            if isspmatrix(X2) and not isspmatrix(X1):
                # we'll need to transpose Y to put X2 first
                order='F'
            else:
                order='C'
            Y = np.empty((m1, m2), dtype=DTYPE, order=order)
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
        X1 : array-like or sparse matrix, shape = (m1, n)
        X2 : array-like or sparse matrix, shape = (m2, n)

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

        X1sp = isspmatrix(X1)
        X2sp = isspmatrix(X2)

        if X1sp:
            X1 = X1.tocsr()
            X1.sort_indices()
        if X2sp:
            X2 = X2.tocsr()
            X2.sort_indices()

        if X1sp and X2sp:
            self._cdist_spsp(X1.data, X1.indices, X1.indptr,
                             X2.data, X2.indices, X2.indptr,
                             X1.shape[1], Y)
        elif X1sp:
            self._cdist_spde(X1.data, X1.indices, X1.indptr, X2, Y)
        elif X2sp:
            self._cdist_spde(X2.data, X2.indices, X2.indptr, X1, Y.T)
        else:
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

        if isspmatrix(X):
            if squareform:
                self._pdist_sp(X.data, X.indices, X.indptr,
                               X.shape[1], Y)
            else:
                self._pdist_sp_compact(X.data, X.indices, X.indptr,
                                       X.shape[1], Y)
        else:
            if squareform:
                self._pdist_c(X, Y)
            else:
                self._pdist_c_compact(X, Y)

        self._cleanup_after_computation()

        return Y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _cdist_cc(DistanceMetric self,
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

    cdef void _cdist_spde(DistanceMetric self,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] X1data,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indices,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indptr,
                          np.ndarray[DTYPE_t, ndim=2, mode='c'] X2,
                          np.ndarray[DTYPE_t, ndim=2, mode='c'] Y):
        # cdist() workhorse. '_spde' means X1 is sparse (csr) and X2 is
        # dense, c-ordered.
        # Arrays are assumed to have:
        #   X1data.shape == n1 
        #   X1indices.shape == n1
        #   X1indptr.shape == m1 + 1
        #   X2.shape = (m2, n)
        #   Y.shape = (m1, m2), where m1 is the number of rows in X1
        cdef Py_ssize_t i1, i2, m1, m2
        cdef int n1, n

        m1 = Y.shape[0]
        m2 = Y.shape[1]
        n = X2.shape[1]
        
        cdef DTYPE_t* pX1 = <DTYPE_t*> X1data.data
        cdef ITYPE_t* pX1i = <ITYPE_t*> X1indices.data
        cdef DTYPE_t* pX2 = <DTYPE_t*> X2.data

        for i1 from 0 <= i1 < m1:
            pX1 = <DTYPE_t*> X1data.data + X1indptr[i1]
            pX1i = <ITYPE_t*> X1indices.data + X1indptr[i1]
            n1 = X1indptr[i1 + 1] - X1indptr[i1]

            pX2 = <DTYPE_t*> X2.data
            for i2 from 0 <= i2 < m2:
                Y[i1, i2] = self.dfunc_spde(pX1, pX1i, n1,
                                            pX2, n,
                                            &self.params, i1, i2)
                pX2 += n

    cdef void _cdist_spsp(DistanceMetric self,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] X1data,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indices,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indptr,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] X2data,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X2indices,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X2indptr,
                          int n,
                          np.ndarray[DTYPE_t, ndim=2, mode='c'] Y):
        # cdist() workhorse. '_spsp' means X1 & X2 are sparse (csr format)
        # Arrays are assumed to have:
        #   X1data.shape == n1 
        #   X1indices.shape == n1
        #   X1indptr.shape == m1 + 1
        #   X1data.shape == n2
        #   X1indices.shape == n2
        #   X1indptr.shape == m2 + 1
        #   n is the dimension of the data (X1.shape[1], X2.shape[1])
        #   Y.shape = (m1, m2), where m1 (m2) is the number of rows in X1 (X2)
        cdef Py_ssize_t i1, i2, m1, m2
        cdef int n1, n2

        m1 = Y.shape[0]
        m2 = Y.shape[1]
        
        cdef DTYPE_t *pX1, *pX2
        cdef ITYPE_t *pX1i, *pX2i

        for i1 from 0 <= i1 < m1:
            pX1 = <DTYPE_t*> X1data.data + X1indptr[i1]
            pX1i = <ITYPE_t*> X1indices.data + X1indptr[i1]
            n1 = X1indptr[i1 + 1] - X1indptr[i1]

            for i2 from 0 <= i2 < m2:
                pX2 = <DTYPE_t*> X2data.data + X2indptr[i2]
                pX2i = <ITYPE_t*> X2indices.data + X2indptr[i2]
                n2 = X2indptr[i2 + 1] - X2indptr[i2]

                Y[i1, i2] = self.dfunc_spsp(pX1, pX1i, n1,
                                            pX2, pX2i, n2, n,
                                            &self.params, i1, i2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _pdist_c(DistanceMetric self,
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

    cdef void _pdist_sp(DistanceMetric self,
                        np.ndarray[DTYPE_t, ndim=1, mode='c'] Xdata,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindices,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindptr,
                        int n,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] Y):
        # pdist() workhorse.  '_sp' means X is sparse csr
        # arrays are assumed to have:
        #   X.shape == (m, n)
        #   Y.shape == (m, m)
        # this is not checked within the function.
        cdef Py_ssize_t i1, i2, m, n1, n2
        m = Y.shape[0]
        
        cdef DTYPE_t *pX1, *pX2
        cdef ITYPE_t *pX1i, *pX2i

        for i1 from 0 <= i1 < m:
            pX1 = <DTYPE_t*> Xdata.data + Xindptr[i1]
            pX1i = <ITYPE_t*> Xindices.data + Xindptr[i1]
            n1 = Xindptr[i1 + 1] - Xindptr[i1]

            for i2 from i1 <= i2 < m:
                pX2 = <DTYPE_t*> Xdata.data + Xindptr[i2]
                pX2i = <ITYPE_t*> Xindices.data + Xindptr[i2]
                n2 = Xindptr[i2 + 1] - Xindptr[i2]

                Y[i1, i2] = Y[i2, i1] = self.dfunc_spsp(pX1, pX1i, n1,
                                                        pX2, pX2i, n2, n,
                                                        &self.params, i1, i2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _pdist_c_compact(DistanceMetric self,
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

    cdef void _pdist_sp_compact(DistanceMetric self,
                                np.ndarray[DTYPE_t, ndim=1, mode='c'] Xdata,
                                np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindices,
                                np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindptr,
                                int n,
                                np.ndarray[DTYPE_t, ndim=1, mode='c'] Y):
        # pdist() workhorse.  '_sp' means X is sparse csr
        #                     '_compact' means Y is in compact form
        # arrays are assumed to have:
        #   X.shape == (m, n)
        #   Y.size == m * (m - 1) / 2
        # this is not checked within the function.
        cdef Py_ssize_t i, i1, i2, n1, n2
        m = Xindptr.shape[0] - 1

        cdef DTYPE_t *pX1, *pX2
        cdef ITYPE_t *pX1i, *pX2i
        
        i = 0
        for i1 from 0 <= i1 < m:
            pX1 = <DTYPE_t*> Xdata.data + Xindptr[i1]
            pX1i = <ITYPE_t*> Xindices.data + Xindptr[i1]
            n1 = Xindptr[i1 + 1] - Xindptr[i1]

            for i2 from i1 < i2 < m:
                pX2 = <DTYPE_t*> Xdata.data + Xindptr[i2]
                pX2i = <ITYPE_t*> Xindices.data + Xindptr[i2]
                n2 = Xindptr[i2 + 1] - Xindptr[i2]

                Y[i] = self.dfunc_spsp(pX1, pX1i, n1,
                                       pX2, pX2i, n2, n,
                                       &self.params, i1, i2)
                i += 1


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

