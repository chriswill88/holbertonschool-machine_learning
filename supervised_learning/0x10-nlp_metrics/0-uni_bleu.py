#!/usr/bin/env python3
"""this module contains a function for task 0"""
import numpy as np


def uni_bleu(references, sentence):
    """this function gives us the unigram bleu score"""
    total = len(sentence)
    reflen = sorted([len(i) for i in references])

    if total not in reflen:
        for i in reflen:
            lref = i
            if total <= i:
                break
    else:
        lref = total

    uniq_ref = {}
    for i in references:
        for x in i:
            if x not in uniq_ref:
                uniq_ref[x] = max([l.count(x) for l in references])

    bp = 1 if total >= lref else np.exp(1 - (lref/total))

    uniq_words = {}
    for i in sentence:
        if i in uniq_ref and i not in uniq_words:
            c = sentence.count(i)
            if c > uniq_ref[i]:
                uniq_words[i] = uniq_ref[i]
            else:
                uniq_words[i] = c
    find = sum(uniq_words.values())

    return (find/total) * bp
