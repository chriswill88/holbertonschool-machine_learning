#!/usr/bin/env python3
import sys
"""this modual holds the function for task 9"""
sys.setrecursionlimit(1501)


def summation_i_squared(n):
    """this function does sqared factorial from n to 0"""
    if not isinstance(n, int) or n <= 0:
        return None
    if (n < 2):
        return 1
    return((n * n) + summation_i_squared(n-1))
