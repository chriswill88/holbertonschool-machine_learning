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
    if not all([isinstance(x, np.ndarray) for x in (X, pi, m, S)]):
        return None, None
    if len(X.shape) != 2 or len(pi.shape) != 1\
            or len(m.shape) != 2 or len(S.shape) != 3:
        return None, None

    p1 = np.array([pdf(X, m[i], S[i]) * pi[i] for i in range(pi.shape[0])])
    s_p = np.sum(p1, 0)
    g = p1/s_p
    return g, np.sum(np.log(s_p))
