#!/usr/bin/env python3
"""This module contains the function for task 0"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """this function creates a bag of word embedding matrix"""
    vect = CountVectorizer(sentences, vocabulary=vocab)
    X = vect.fit_transform(sentences)
    return vect.get_feature_names(), X.toarray()
