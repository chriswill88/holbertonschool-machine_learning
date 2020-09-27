#!/usr/bin/env python3
"""this module contains a function for task 4"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Positional Encoding"""
    pe = np.arange(max_seq_len)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(dm)//2)) / np.float32(dm)
        )[np.newaxis, :]
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe
