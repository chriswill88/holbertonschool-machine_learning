#!/usr/bin/env python3
"""this function calculates the persicion"""
import numpy as np


def precision(confusion):
    """calculates the persicion for each class in the matrix"""
    classes = len(confusion)
    per = np.zeros((classes,))
    for i in range(len(confusion)):
        num = confusion[i][i]

        per[i] = num/(np.sum(confusion[:, i]))
    return per
