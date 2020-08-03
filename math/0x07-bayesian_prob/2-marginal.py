#!/usr/bin/env python3
"""this module contains the function used in task 1"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    @x: the number of patients that develop severe side effects
    @n: the total number of patients observed
    @p: a 1D numpy.ndarray containing the various hypothetical probabilities
     of developing severe side effects
    @Pr is a 1D numpy.ndarray containing the prior beliefs of P

    This function calculates the likelihood of obtaining the data (x, n) given
     various probabilities
    to solve this the pmf for binomial distributions was used.

    This function calculates the marginal probability
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1.):
        raise ValueError("Pr must sum to 1")
    fact = np.math.factorial
    c = fact(n)/(fact(n-x)*fact(x))
    return np.sum(c*((P**x)*((1-P)**(n-x))) * Pr)
