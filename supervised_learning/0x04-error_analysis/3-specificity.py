#!/usr/bin/env python3
"""this function calculates the specificity"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in the matrix"""
    classes = len(confusion)
    total = 0
    per = np.zeros((classes,))
    total = sum(sum(confusion))

    for i in range(len(confusion)):
        fp = sum(confusion[:, i]) - confusion[i][i]
        tn = total - confusion[i][i]
        per[i] = tn/(fp + tn)
    return per
