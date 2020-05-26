#!/usr/bin/env python3
"""this function calculates the specificity"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in the matrix"""
    classes = len(confusion)
    truth = 0
    per = np.zeros((classes,))
    # tru_confusion = np.zeros((2, 2))

    total = np.sum(confusion)
    for i in range(len(confusion)):
        tp = confusion[i][i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i]) - tp
        tn = total - tp - fp - fn

        # tru_confusion[0][0] = tp
        # tru_confusion[1][0] = fp
        # tru_confusion[0][1] = fn
        # tru_confusion[1][1] = tn

        per[i] = tn/(tn + fp)

    return per
