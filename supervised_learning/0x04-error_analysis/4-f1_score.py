#!/usr/bin/env python3
"""this function calculates the f1 score"""
import numpy as np


def f1_score(confusion):
    """calculates the f1 score for each class in the matrix"""
    classes = len(confusion)
    truth = 0
    per = np.zeros((classes,))
    rec = np.zeros((classes,))

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

        per[i] = tp / (tp + fp)
        rec[i] = tp/(tp + fn)
    return 2*((per*rec)/(per+rec))
