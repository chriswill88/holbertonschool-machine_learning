#!/usr/bin/env python3
"""This module contains the function for task 1"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """this function creates a bag of word embedding matrix"""
    vect = TfidfVectorizer(sentences, vocabulary=vocab)
    X = vect.fit_transform(sentences)
    return X.toarray(), vect.get_feature_names()
