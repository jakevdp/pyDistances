import numpy as np
from scipy import sparse

# safe sparse dot: taken from scikit-learn utilities
def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly"""
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


# safe asarray: taken from scikit-learn utilities
def safe_asarray(X, dtype=None, order=None):
    """Convert X to an array or sparse matrix.

    Prevents copying X when possible; sparse matrices are passed through."""
    if not sparse.issparse(X):
        X = np.asarray(X, dtype, order)
    return X


def corr(X1, X2=None):
    """Compute the correlation matrix

    To machine precision, this is designed so that:

    corr(X) == np.cov(X.T)
    corr(X1, X2) == np.cov(np.hstack((X1.T, X2.T)))

    X1/X2 can be dense or sparse, and the computation uses memory efficiently
    but may be numerically unstable if the mean is very large.
    """
    X1 = safe_asarray(X1)
    N, D = X1.shape
    use_X2 = (X2 is not None)

    if use_X2:
        X2 = safe_asarray(X2)
        N2, D2 = X2.shape
        assert D2 == D
        N += N2

    # we will stack X1 and X2, compute the mean mu = X12.mean(0)
    # then return dot((X12 - mu).T, (X12 - mu)) / N
    # but do this in a memory-efficient way.

    X1sum = np.asarray(X1.sum(0))
    mu = X1sum

    if use_X2:
        X2sum = np.asarray(X2.sum(0))
    else:
        X2sum = 0

    mu = np.atleast_2d(mu + X2sum)
    mu /= N

    C = np.zeros((D, D), dtype=float)
    C += safe_sparse_dot(X1.T, X1, dense_output=True)

    if use_X2:
        C += safe_sparse_dot(X2.T, X2, dense_output=True)
        muX = mu.T * (X1sum + X2sum)
    else:
        muX = mu.T * X1sum

    C -= muX
    C -= muX.T

    C += N * np.dot(mu.T, mu)
    
    C /= (N - 1)

    return C

def safe_sparse_square(X):
    """return the elementwise product of x and y"""
    if sparse.isspmatrix(X):
        return X.multiply(X)
    else:
        return np.multiply(X, X)

def column_var(X1, X2=None):
    use_X2 = (X2 is not None)
    if sparse.isspmatrix(X1) or sparse.isspmatrix(X2):
        X2sum = np.asarray(safe_sparse_square(X1).sum(0)).ravel()
        Xsum = np.asarray(X1.sum(0)).ravel()
        N = X1.shape[0]
        
        if use_X2:
            X2sum += np.asarray(safe_sparse_square(X2).sum(0)).ravel()
            Xsum += np.asarray(X2.sum(0)).ravel()
            N += X2.shape[0]

        X2sum = np.asarray(X2sum).ravel()
        Xsum = np.asarray(Xsum).ravel()

        return (X2sum - Xsum ** 2 / N) / float(N - 1)
    else:
        if use_X2:
            X = np.vstack((X1, X2))
        else:
            X = np.asarray(X1)
        return X.var(axis=0, ddof=1)

def _norms(X):
    if sparse.isspmatrix(X):
        return np.sqrt(np.asarray((X.multiply(X)).sum(1),
                                  dtype=X.dtype, order='C'))
    else:
        return np.sqrt(np.asarray((X ** 2).sum(1),
                                  dtype=X.dtype, order='C'))


def _centered(X):
    return X - X.mean(1).reshape((-1, 1))
