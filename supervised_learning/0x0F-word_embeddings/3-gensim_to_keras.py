#!/usr/bin/env python3
"""This module contains the function for task 3"""


def gensim_to_keras(model):
    """gensim_to_keras - this function return an embedded layer"""
    return model.wv.get_keras_embedding(True)
