#!/usr/bin/env python3
"""this module contains a function for task 1"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """this function gives us the unigram bleu score"""
    sen_leng = len(sentence)
    reflen = sorted([len(i) for i in references])

    uniq_words = {}
    for i in range(sen_leng):
        if i + n <= sen_leng:
            uniq_words[str(sentence[i:i+n])] = 0
    total = len(uniq_words)

    if total not in reflen:
        for i in reflen:
            lref = i
            if total <= i:
                break
    else:
        lref = total

    uniq_ref = {}
    for i in references:
        length = len(i)
        for x in range(length):
            var = str(i[x:x+n])
            if x + n <= len(i):
                if var not in uniq_ref:
                    uniq_ref[var] = 1
                else:
                    uniq_ref[var] += 1

    for phrase, freq in uniq_words.items():
        if phrase in uniq_ref.keys():
            if uniq_ref[phrase] > uniq_words[phrase]:
                uniq_words[phrase] = uniq_ref[phrase]
            else:
                uniq_words[phrase] += 1
    find = sum(uniq_words.values())
    bp = 1 if total >= lref else np.exp(1 - (lref/sen_leng))

    return (find/total) * bp
