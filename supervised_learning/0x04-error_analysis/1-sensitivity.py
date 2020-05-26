#!/usr/bin/env python3
"""this function calculates the sensitivity"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in the matrix"""
    classes = len(confusion)
    sen = np.zeros((classes,))
    for i in range(len(confusion)):
        num = confusion[i][i]
        sen[i] = num/np.sum(confusion[i])
    return sen
