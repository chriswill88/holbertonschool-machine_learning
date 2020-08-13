#!/usr/bin/env python3
"""This modual contains the function for task 0"""
import numpy as np


def markov_chain(P, s, t=1):
    """This function returns the state after an amount of iterations"""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if n1 != n2:
        return None
    if not isinstance(
            s, np.ndarray) or len(s.shape) != 2 or s.shape[0] != (1, n1):
        return None
    if t < 0:
        return None
    for i in range(t):
        s = s @ P
    return s
