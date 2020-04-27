#!/usr/bin/env python3
"""this modual holds the function for task 17"""


def poly_integral(poly, C=0):
    """returns the new coeffiecients for an integral"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    lis = [C]

    if len(poly) == 1:
        return lis

    for i in range(len(poly)):
        if i > 0:
            if not poly[i] % (i + 1):
                new = int(poly[i]/(i + 1))
            else:
                new = poly[i]/(i + 1)
            lis.append(new)
        else:
            lis.append(poly[i])
    return lis
