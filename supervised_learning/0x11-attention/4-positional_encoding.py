#!/usr/bin/env python3
"""this module contains a function for task 4"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Positional Encoding"""
    pe = np.zeros((max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            pe[pos, i] = \
                np.sin(pos / (10000 ** ((2 * i)/dm)))
            pe[pos, i + 1] = \
                np.cos(pos / (10000 ** ((2 * (i + 1))/dm)))
    return pe
