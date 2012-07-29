#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

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
ctypedef DTYPE_t (*distfunc)(DTYPE_t[:, ::1], ITYPE_t,
                             DTYPE_t[:, ::1], ITYPE_t)

cdef DTYPE_t euclidean_dist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                            DTYPE_t[:, ::1] X2, ITYPE_t i2):
    cdef ITYPE_t k, m = X1.shape[1]
    cdef DTYPE_t d, rdist = 0

    for k in range(m):
        d = X1[i1, k] - X2[i2, k]
        rdist += d * d

    return sqrt(rdist)


cdef DTYPE_t raw_dist(DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
    return 0.0


#----------------------------------------------------------------------
# forward-declare distance metric class
cdef class DistanceMetric

cdef class Pairwise:
    cdef DistanceMetric dm
    def __init__(self, metric, **kwargs):
        if isinstance(metric, DistanceMetric):
            self.dm = metric
        elif metric == "raw":
            self.dm = DistanceMetric()
        elif metric == "euclidean":
            self.dm = EuclideanDistance()
        else:
            raise ValueError("unrecognized metric: %s" % str(metric))
        
    def cdist(self, DTYPE_t[:, ::1] X1, DTYPE_t[:, ::1] X2):
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 and X2 must have the same final dimension")

        cdef DTYPE_t[:, ::1] res = np.empty((X1.shape[0], X2.shape[0]),
                                            dtype=DTYPE, order='c')
        self._cdist(X1, X2, res)
        return res

    cdef void _cdist(self,
                     DTYPE_t[:, ::1] X1,
                     DTYPE_t[:, ::1] X2,
                     DTYPE_t[:, ::1] res):
        cdef ITYPE_t i1, i2
        cdef DTYPE_t d

        #cdef distfunc func = self.dm.dist
        
        for i1 in range(res.shape[0]):
            for i2 in range(res.shape[1]):
                #d = euclidean_dist(X1, i1, X2, i2)
                #d = raw_dist(X1, i1, X2, i2)
                d = self.dm.dist(X1, i1, X2, i2)
                #d = func(X1, i1, X2, i2)
                res[i1, i2] = d

#----------------------------------------------------------------------
"""
cdef class DistanceMetric:
    cdef distfunc dist
    cdef distfunc rdist
    def __init__(self):
        dist = raw_dist
        rdist = raw_dist

cdef class EuclideanDistance(DistanceMetric):
    def __init__(self):
        self.dist = euclidean_dist
        self.rdist = euclidean_dist

"""
cdef class DistanceMetric:
    cdef DTYPE_t dist(self,
                      DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        return 0.0

    cdef DTYPE_t rdist(self,
                      DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        return 0.0

    cdef DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist

    cdef DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return rdist


cdef class EuclideanDistance(DistanceMetric):
    cdef DTYPE_t dist(self,
                      DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        #return sqrt(self.rdist(X1, i1, X2, i2))
        cdef ITYPE_t m = X1.shape[1]
        cdef ITYPE_t k
        cdef DTYPE_t d, rdist = 0

        for k in range(m):
            d = X1[i1, k] - X2[i2, k]
            rdist += d * d

        return sqrt(rdist)

    cdef DTYPE_t rdist(self,
                      DTYPE_t[:, ::1] X1, ITYPE_t i1,
                      DTYPE_t[:, ::1] X2, ITYPE_t i2):
        cdef ITYPE_t m = X1.shape[1]
        cdef ITYPE_t k
        cdef DTYPE_t d, rdist = 0

        for k in range(m):
            d = X1[i1, k] - X2[i2, k]
            rdist += d * d

        return rdist

    cdef DTYPE_t dist_to_rdist(self, DTYPE_t dist):
        return dist * dist

    cdef DTYPE_t rdist_to_dist(self, DTYPE_t rdist):
        return sqrt(rdist)

