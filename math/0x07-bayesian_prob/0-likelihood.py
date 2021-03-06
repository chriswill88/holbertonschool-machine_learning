#!/usr/bin/env python3
"""this module contains the function used in task 0"""
import numpy as np


def likelihood(x, n, P):
    """
    @x: the number of patients that develop severe side effects
    @n: the total number of patients observed
    @p: a 1D numpy.ndarray containing the various hypothetical probabilities
     of developing severe side effects

    This function calculates the likelihood of obtaining the data (x, n) given
     various probabilities
    to solve this the pmf for binomial distributions was used.

    this function returns a 1D array of likelihoods
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if len(np.where(P < 0)[0]) or len(np.where(P > 1)[0]):
        raise ValueError("All values in P must be in the range [0, 1]")

    fact = np.math.factorial
    c = fact(n)/(fact(n-x)*fact(x))
    return c*((P**x)*((1-P)**(n-x)))
