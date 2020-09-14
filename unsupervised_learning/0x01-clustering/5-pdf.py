#!/usr/bin/env python3
"""This function computes the pdf for gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """Calc the pdf of a gaussian distribution"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(X.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(X.shape) != 2:
        return None

    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n = m.shape[0]
    Sigma_det = np.linalg.det(S)
    Sigma_inv = np.linalg.inv(S)
    # front part
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-m)T.S-1.(x-m) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', X-m, Sigma_inv, X-m)
    pdf = np.exp(-fac / 2) / N
    return np.where(pdf < 1e-300, 1e-300, pdf)
