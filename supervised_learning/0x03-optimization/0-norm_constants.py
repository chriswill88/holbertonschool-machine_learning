#!/usr/bin/env python3
"""this modual is for task 0"""
import numpy as np


def normalization_constants(X):
    """calulates the normalizaton constants"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
