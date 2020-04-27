#!/usr/bin/env python3
"""this modual holds the function for task 17"""


def poly_integral(poly, C=0):
    """returns the new coeffiecients for an integral"""
    lis = [C]

    for i in range(len(poly)):
        if i > 0:
            lis.append(poly[i]/(i + 1))
        else:
            lis.append(poly[i])
    return lis
