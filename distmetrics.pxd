cimport numpy as np

# C data types (corresponding python types should be used in file)
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t
cdef enum:
    DTYPECODE = np.NPY_FLOAT64
    ITYPECODE = np.NPY_INT32

###############################################################################
# Define data structures needed for distance calculations

cdef struct mahalanobis_info:
    DTYPE_t* work_buffer  # pointer to buffer of size n
    Py_ssize_t n          # size of arrays
    DTYPE_t* VI           # pointer to buffer of size n * n

cdef struct minkowski_info:
    Py_ssize_t n  # size of array
    DTYPE_t p     # specifies p-norm
    DTYPE_t* w    # pointer to buffer of size n

cdef struct seuclidean_info:
    Py_ssize_t n   # size of array
    DTYPE_t* V     # pointer to buffer of size n

cdef struct cosine_info:
    int precomputed_norms    # flag to record whether norms are precomputed
    DTYPE_t *norms1, *norms2 # precomputed norms of vectors

cdef struct correlation_info:
    int precomputed_data      # flag to record whether data is pre-centered
                              #  and norms are pre-computed
    DTYPE_t *x1, *x2          # precentered data vectors
    DTYPE_t *norms1, *norms2  # precomputed norms of vectors

cdef struct user_info:
    void* func

# general distance data structure.  We use a union because
# different distance metrics require different ancillary information,
# and we only need memory allocated for one of the structures.
cdef union dist_params:
    minkowski_info minkowski
    mahalanobis_info mahalanobis
    seuclidean_info seuclidean
    cosine_info cosine
    correlation_info correlation
    user_info user

###############################################################################
# Function pointer types

# define a pointer to a generic distance function (dense-dense)
ctypedef DTYPE_t (*dist_func)(DTYPE_t*, DTYPE_t*, int,
                              dist_params*, int, int)

# define a pointer to a generic distance function (sparse-dense)
ctypedef DTYPE_t (*dist_func_spde)(DTYPE_t*, ITYPE_t*, int, DTYPE_t*, int,
                                   dist_params*, int, int)

# define a pointer to a generic distance function (sparse-sparse)
ctypedef DTYPE_t (*dist_func_spsp)(DTYPE_t*, ITYPE_t*, int,
                                   DTYPE_t*, ITYPE_t*, int, int,
                                   dist_params*, int, int)

# pointer to a generic distance conversion function
ctypedef DTYPE_t (*dist_conv_func)(DTYPE_t, dist_params*)


###############################################################################
# DistanceMetric class

cdef class DistanceMetric(object):
    # initialization information
    cdef object metric
    cdef object init_kwargs

    # C attributes: information about and access to distance functions
    cdef dist_params params
    cdef dist_func dfunc
    cdef dist_func reduced_dfunc
    
    cdef dist_func_spde dfunc_spde
    cdef dist_func_spde reduced_dfunc_spde

    cdef dist_func_spsp dfunc_spsp
    cdef dist_func_spsp reduced_dfunc_spsp

    cdef dist_conv_func dist_to_reduced
    cdef dist_conv_func reduced_to_dist

    # array attributes used for various distance measures
    # note: some of these could be combined for a smaller memory footprint,
    #       but for clarity in reading the code we use separate objects.
    cdef np.ndarray mahalanobis_VI     # - inverse covariance matrix of data
    cdef np.ndarray seuclidean_V       # - variance array of data
    cdef np.ndarray minkowski_w        # - weights for weighted minkowski
    cdef np.ndarray norms1             # - precomputed norms, used for cosine
    cdef np.ndarray norms2             #   and correlation distances
    cdef np.ndarray precentered_data1  # - pre-centered data, used for
    cdef np.ndarray precentered_data2  #   correlation distance
    cdef np.ndarray work_buffer        # - work buffer

    # flag for computation
    cdef int learn_params_from_data

    # workhorse routines for efficient distance computation
    cdef void _cdist_cc(DistanceMetric self,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] X1,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] X2,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] Y)

    cdef void _cdist_spde(DistanceMetric self,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] X1data,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indices,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indptr,
                          np.ndarray[DTYPE_t, ndim=2, mode='c'] X2,
                          np.ndarray[ITYPE_t, ndim=2, mode='c'] Y)

    cdef void _cdist_spsp(DistanceMetric self,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] X1data,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indices,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X1indptr,
                          np.ndarray[DTYPE_t, ndim=1, mode='c'] X2data,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X2indices,
                          np.ndarray[ITYPE_t, ndim=1, mode='c'] X2indptr,
                          int n,
                          np.ndarray[ITYPE_t, ndim=2, mode='c'] Y)

    cdef void _pdist_c(DistanceMetric self,
                       np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                       np.ndarray[DTYPE_t, ndim=2, mode='c'] Y)

    cdef void _pdist_c_compact(self,
                               np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] Y)

    cdef void _pdist_sp(DistanceMetric self,
                        np.ndarray[DTYPE_t, ndim=1, mode='c'] Xdata,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindices,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindptr,
                        int n,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] Y)

    cdef void _pdist_sp_compact(self,
                                np.ndarray[DTYPE_t, ndim=1, mode='c'] Xdata,
                                np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindices,
                                np.ndarray[ITYPE_t, ndim=1, mode='c'] Xindptr,
                                int n,
                                np.ndarray[DTYPE_t, ndim=1, mode='c'] Y)
