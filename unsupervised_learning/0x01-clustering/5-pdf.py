#!/usr/bin/env python3
"""This function computes the pdf for gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """Calc the pdf of a gaussian distribution"""
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
