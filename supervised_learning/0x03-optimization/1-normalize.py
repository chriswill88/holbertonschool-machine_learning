#!/usr/bin/env python3
"""this modual holds the function for task 1"""


def normalize(X, m, s):
    """normalizes data"""
    X -= m
    X /= s
    return X
