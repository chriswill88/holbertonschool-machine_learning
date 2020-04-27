#!/usr/bin/env python3
"""this modual holds the function for task 9"""


def summation_i_squared(n):
    """this function does sqared factorial from n to 0"""
    if n <= 0 or not isinstance(n, int):
        return None
    summ = list(map((lambda x: x * x), range(n + 1)))
    return sum(summ)
 
