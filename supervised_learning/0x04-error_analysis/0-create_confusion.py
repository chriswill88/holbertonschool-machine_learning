#!/usr/bin/env python3
"""this modual contains the function for task 0"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    classes = labels.shape[1]
    conf = np.zeros((classes, classes))

    for i in range(len(labels)):
        if labels[i] is logits[i]:
            conf[i][i] += 1
        else:
            log = np.where(logits[i])[0][0]
            tru = np.where(labels[i])[0][0]

            conf[tru][log] += 1
    return conf
