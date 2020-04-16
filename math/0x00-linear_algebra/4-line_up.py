#!/usr/bin/env python3
"""
    In this modual there is the
    add_array function for task 4
"""


def add_arrays(arr1, arr2):
    """
        add_array - does element wise addition
    """
    if (len(arr1) != len(arr2)):
        return (None)
    lis = []
    for i in range(len(arr1)):
        lis.append(arr1[i] + arr2[i])
    return lis
