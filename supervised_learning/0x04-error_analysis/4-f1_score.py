#!/usr/bin/env python3
"""this function calculates the f1 score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the f1 score for each class in the matrix"""
    per = precision(confusion)
    rec = sensitivity(confusion)
    return 2*((per*rec)/(per+rec))
