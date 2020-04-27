#!/usr/bin/env python3
"""this modual holds the function for task 9"""


def summation_i_squared(n):
    """this function does sqared factorial from n to 0"""
    if n <= 0 or not isinstance(n, int):
        return None
    if (n < 2):
        return 1
    return((n * n) + summation_i_squared(n - 1))
