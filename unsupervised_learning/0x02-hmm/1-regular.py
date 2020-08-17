#!/usr/bin/env python3
"""this module contains the function for task 1"""
import numpy as np


def regular(P):
    """regular - determines steady state probability of a regular markov chain"""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if not P.all() > 0:
        return None
    state_matrix = np.ones((1, P.shape[0]))/P.shape[0]
    prev = None
    for i in range(100):
        comparison = prev == state_matrix
        if prev is not None and comparison.all():
            return state_matrix
        prev = state_matrix
        state_matrix = np.matmul(state_matrix, P)
    return None