#!/usr/bin/env python3
"""This function computes the pdf for gaussian distribution"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    expectation step in the EM algorithm for a GMM:
    @X is a numpy.ndarray of shape (n, d) containing the data set
    @pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    @m is a numpy.ndarray of shape (k, d) containing the centroid means for
     each cluster
    @S is a numpy.ndarray of shape (k, d, d) containing the covariance
     matrices for each cluster
    """
    print("pi is", pi)
    if not isinstance(X, np.ndarray) or not isinstance(pi, np.ndarray)\
            or not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray):
        return None, None

    if len(X.shape) != 2 or len(pi.shape) != 1\
            or len(m.shape) != 2 or len(S.shape) != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    try:
        assert d == m.shape[1]
        assert d == S.shape[1] and S.shape[1] == S.shape[2]
        assert k == m.shape[0] and k == S.shape[0]
    except AssertionError:
        return None, None
    print("here")
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    p1 = np.array([pdf(X, m[i], S[i]) * pi[i] for i in range(pi.shape[0])])
    s_p = np.sum(p1, 0)
    g = p1/s_p
    return g, np.sum(np.log(s_p))
