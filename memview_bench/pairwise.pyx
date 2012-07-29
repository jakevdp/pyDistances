#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt

# Data type
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# Index type
ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

#----------------------------------------------------------------------

def pairwise1(np.ndarray[DTYPE_t, ndim=2, mode='c'] X1,
              np.ndarray[DTYPE_t, ndim=2, mode='c'] X2):
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same final dimension")

    cdef np.ndarray res = np.empty((X1.shape[0], X2.shape[0]),
                                   dtype=DTYPE, order='c')

    _pairwise1(<DTYPE_t*> X1.data,
               <DTYPE_t*> X2.data,
               <DTYPE_t*> res.data,
               X1.shape[0], X2.shape[0], X1.shape[1])
    
    return res

cdef void _pairwise1(DTYPE_t* X1, DTYPE_t* X2, DTYPE_t* res,
                     ITYPE_t N1, ITYPE_t N2, ITYPE_t D):
    cdef ITYPE_t i1, i2, k
    cdef DTYPE_t d, dist
    
    for i1 in range(N1):
        for i2 in range(N2):
            dist = 0
            for k in range(D):
                d = X1[i1 * D + k] - X2[i2 * D + k]
                dist += d * d
            res[i1 * N2 + i2] = sqrt(dist)

#----------------------------------------------------------------------

def pairwise2(DTYPE_t[:, ::1] X1, DTYPE_t[:, ::1] X2):
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same final dimension")

    cdef np.ndarray res = np.empty((X1.shape[0], X2.shape[0]),
                                   dtype=DTYPE, order='c')

    _pairwise2(X1, X2, res)
    
    return res


cdef void _pairwise2(DTYPE_t[:, ::1] X1,
                     DTYPE_t[:, ::1] X2,
                     DTYPE_t[:, ::1] res):
    cdef ITYPE_t i1, i2, k, N1, N2, D
    cdef DTYPE_t d, dist

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    D = X1.shape[1]
    
    for i1 in range(N1):
        for i2 in range(N2):
            dist = 0
            for k in range(D):
                d = X1[i1, k] - X2[i2, k]
                dist += d * d
            res[i1, i2] = sqrt(dist)

#----------------------------------------------------------------------
    
def pairwise3(np.ndarray[DTYPE_t, ndim=2, mode='c'] X1,
              np.ndarray[DTYPE_t, ndim=2, mode='c'] X2):
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same final dimension")

    cdef np.ndarray res = np.empty((X1.shape[0], X2.shape[0]),
                                   dtype=DTYPE, order='c')

    _pairwise3(<DTYPE_t*> X1.data,
               <DTYPE_t*> X2.data,
               <DTYPE_t*> res.data,
               X1.shape[0], X2.shape[0], X1.shape[1])
    
    return res

cdef void _pairwise3(DTYPE_t* X1, DTYPE_t* X2, DTYPE_t* res,
                     ITYPE_t N1, ITYPE_t N2, ITYPE_t D):
    cdef ITYPE_t i1, i2
    for i1 in range(N1):
        for i2 in range(N2):
            res[i1 * N2 + i2] = euclidean_dist3(X1 + i1 * D, X2 + i2 * D, D)

cdef DTYPE_t euclidean_dist3(DTYPE_t* x1, DTYPE_t* x2, ITYPE_t D):
    cdef ITYPE_t k
    cdef DTYPE_t d, rdist = 0
    for k in range(D):
        d = x1[k] - x2[k]
        rdist += d * d
    return sqrt(rdist)

#----------------------------------------------------------------------

def pairwise4(DTYPE_t[:, ::1] X1, DTYPE_t[:, ::1] X2):
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same final dimension")

    cdef np.ndarray res = np.empty((X1.shape[0], X2.shape[0]),
                                   dtype=DTYPE, order='c')

    _pairwise4(X1, X2, res)
    
    return res


cdef void _pairwise4(DTYPE_t[:, ::1] X1,
                     DTYPE_t[:, ::1] X2,
                     DTYPE_t[:, ::1] res):
    cdef ITYPE_t i1, i2, N1, N2, D

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    D = X1.shape[1]
    
    for i1 in range(N1):
        for i2 in range(N2):
            res[i1, i2] = euclidean_dist4(X1[i1, :], X2[i2, :])


cdef DTYPE_t euclidean_dist4(DTYPE_t[::1] x1, DTYPE_t[::1] x2):
    cdef ITYPE_t i, m = x1.shape[0]
    cdef DTYPE_t d, rdist = 0

    for i in range(m):
        d = x1[i] - x2[i]
        rdist += d * d

    return sqrt(rdist)

#----------------------------------------------------------------------
    
def pairwise5(np.ndarray[DTYPE_t, ndim=2, mode='c'] X1,
              np.ndarray[DTYPE_t, ndim=2, mode='c'] X2):
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same final dimension")

    cdef np.ndarray res = np.empty((X1.shape[0], X2.shape[0]),
                                   dtype=DTYPE, order='c')

    _pairwise5(<DTYPE_t*> X1.data,
               <DTYPE_t*> X2.data,
               <DTYPE_t*> res.data,
               X1.shape[0], X2.shape[0], X1.shape[1])
    
    return res

cdef void _pairwise5(DTYPE_t* X1, DTYPE_t* X2, DTYPE_t* res,
                     ITYPE_t N1, ITYPE_t N2, ITYPE_t D):
    cdef ITYPE_t i1, i2
    for i1 in range(N1):
        for i2 in range(N2):
            res[i1 * N2 + i2] = euclidean_dist5(X1, X2, i1, i2, D)

cdef DTYPE_t euclidean_dist5(DTYPE_t* X1, DTYPE_t* X2,
                             ITYPE_t i1, ITYPE_t i2, ITYPE_t D):
    cdef ITYPE_t k
    cdef DTYPE_t d, rdist = 0
    for k in range(D):
        d = X1[i1 * D + k] - X2[i2 * D + k]
        rdist += d * d
    return sqrt(rdist)

#----------------------------------------------------------------------

def pairwise6(DTYPE_t[:, ::1] X1, DTYPE_t[:, ::1] X2):
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same final dimension")

    cdef np.ndarray res = np.empty((X1.shape[0], X2.shape[0]),
                                   dtype=DTYPE, order='c')

    _pairwise6(X1, X2, res)
    
    return res


cdef void _pairwise6(DTYPE_t[:, ::1] X1,
                     DTYPE_t[:, ::1] X2,
                     DTYPE_t[:, ::1] res):
    cdef ITYPE_t i1, i2, N1, N2, D

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    D = X1.shape[1]
    
    for i1 in range(N1):
        for i2 in range(N2):
            res[i1, i2] = euclidean_dist6(X1, i1, X2, i2)


cdef DTYPE_t euclidean_dist6(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                             DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef ITYPE_t k, m = X1.shape[1]
    cdef DTYPE_t d, rdist = 0

    for k in range(m):
        d = X1[i1, k] - X2[i2, k]
        rdist += d * d

    return sqrt(rdist)
