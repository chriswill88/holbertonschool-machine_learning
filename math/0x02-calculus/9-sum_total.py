#!/usr/bin/env python3
"""this modual holds the function for task 9"""


def summation_i_squared(n):
    """this function does sqared factorial from n to 0"""
    if n < 0:
        return None
    if (n < 1):
        return 0
    return((n * n) + summation_i_squared(n - 1))
