cimport numpy as np

# C data types (corresponding python types should be used in file)
ctypedef np.float64_t DTYPE_t

###############################################################################
# Define data structures needed for distance calculations

# data structure used for mahalanobis distance
cdef struct mahalanobis_info:
    Py_ssize_t n          # size of arrays
    DTYPE_t* VI           # pointer to buffer of size n * n
    DTYPE_t* work_buffer  # pointer to buffer of size n

# data structure used for (weighted) minkowski distance
cdef struct minkowski_info:
    Py_ssize_t n  # size of array
    DTYPE_t p     # specifies p-norm
    DTYPE_t* w    # pointer to buffer of size n

# data structure used for standardized euclidean distance
cdef struct seuclidean_info:
    Py_ssize_t n   # size of array
    DTYPE_t* V     # pointer to buffer of size n

# data structure used for cosine distance
cdef struct cosine_info:
    int precomputed_norms    # flag to record whether norms are precomputed
    DTYPE_t *norms1, *norms2 # precomputed norms of vectors

# data structure used for correlation distance
cdef struct correlation_info:
    int precomputed_data      # flag to record whether data is pre-centered
                              #  and norms are pre-computed
    DTYPE_t *x1, *x2          # precentered data vectors
    DTYPE_t *norms1, *norms2  # precomputed norms of vectors

# data structure for user-defined metric
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

# define a pointer to a generic distance function.
ctypedef DTYPE_t (*dist_func)(DTYPE_t*, DTYPE_t*, Py_ssize_t,
                              dist_params*, Py_ssize_t, Py_ssize_t)

# distance conversion function
ctypedef DTYPE_t (*dist_conv_func)(DTYPE_t, dist_params*)

###############################################################################
# Utility routines

# from a distance function, obtain the reduced distance function
cdef dist_func get_reduced_dfunc(dist_func)

# from a distance function, obtain the function to convert from
#  a true distance measurement to its reduced form
cdef dist_conv_func get_dist_to_reduced(dist_func)

# from a distance function, obtain the function to convert from
#  a reduced-form distance measurement to the true distance
cdef dist_conv_func get_reduced_to_dist(dist_func)

###############################################################################
# DistanceMetric class

cdef class DistanceMetric(object):
    # C attributes which store information about the distance function
    cdef dist_params params
    cdef dist_func dfunc

    # array attributes used for various distance measures
    # note: some of these could be combined for a smaller memory footprint,
    #       but for clarity in reading the code we use separate objects.
    cdef np.ndarray mahalanobis_VI     # inverse covariance matrix of data
    cdef np.ndarray seuclidean_V       # variance array of data
    cdef np.ndarray minkowski_w        # weights for weighted minkowski
    cdef np.ndarray norms1             # precomputed norms, used for cosine
    cdef np.ndarray norms2             #  and correlation distances
    cdef np.ndarray precentered_data1  # pre-centered data, used for
    cdef np.ndarray precentered_data2  #  correlation distance
    cdef np.ndarray work_buffer        # work buffer

    # flags for computation
    cdef int learn_params_from_data

    cdef void _cdist_cc(self,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] X1,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] X2,
                        np.ndarray[DTYPE_t, ndim=2, mode='c'] Y)

    cdef void _pdist_c(self,
                       np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                       np.ndarray[DTYPE_t, ndim=2, mode='c'] Y)

    cdef void _pdist_c_compact(self,
                               np.ndarray[DTYPE_t, ndim=2, mode='c'] X,
                               np.ndarray[DTYPE_t, ndim=1, mode='c'] Y)
