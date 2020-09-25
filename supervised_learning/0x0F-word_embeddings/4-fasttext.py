#!/usr/bin/env python3
"""this module contains the function for task 4"""
from gensim.models import FastText


def fasttext_model(
    sentences, size=100, min_count=5, negative=5, window=5,
        cbow=True, iterations=5, seed=0, workers=1):
    """This function sets up fasttext model"""
    cbow = 0 if cbow == 1 else 1
    model = FastText(
        sentences, size=size, min_count=min_count,
        negative=negative, window=window, sg=cbow, iter=iterations,
        seed=seed, workers=workers)

    return model
