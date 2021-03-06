#!/usr/bin/env python3
"""This modual holds the function for Task 10 """


def poly_derivative(poly):
    """figures out the new coefficients for derivatives"""
    lis = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    for i in range(len(poly)):
        if i > 0:
            lis.append(poly[i] * i)
    return lis
