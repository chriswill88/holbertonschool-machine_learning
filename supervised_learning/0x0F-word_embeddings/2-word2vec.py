#!/usr/bin/env python3
"""This module contains the function for task 2"""
from gensim.models import Word2Vec


def word2vec_model(
    sentences, size=100, min_count=5, window=5, negative=5,
        cbow=True, iterations=5, seed=0, workers=1):
    """This function trains a word2vec function"""
    cbow = 0 if cbow == 1 else 1
    model = Word2Vec(
        sentences, size=size, min_count=min_count,
        window=window, negative=negative, iter=iterations,
        seed=seed, workers=workers, sg=cbow)

    return model
