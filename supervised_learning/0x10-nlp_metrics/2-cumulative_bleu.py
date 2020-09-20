#!/usr/bin/env python3
"""this module contains a function for task 2"""
import numpy as np


def nparser(sentence, n):
    """nparser - parses sentence in to n partitions"""
    uniq_words = []
    for i in range(len(sentence)):
        if i + n <= len(sentence):
            uniq_words.append(str(sentence[i:i+n]))
    return uniq_words


def ngram_bleu(references, sentence, n):
    """this function gives us the unigram bleu score"""
    total = len(sentence)
    lref = min([len(i) for i in references])

    parsed_sent = nparser(sentence, n)
    # print(parsed_sent)

    parsed_tot = len(parsed_sent)

    parsed_ref = []
    for i in references:
        parsed_ref.append(nparser(i, n))
    # print(parsed_ref)

    uniq_ref = {}
    for i in parsed_ref:
        for x in i:
            if x not in uniq_ref:
                uniq_ref[x] = max([l.count(x) for l in parsed_ref])
    # print("ur", uniq_ref)

    uniq_words = {}
    for phrase in parsed_sent:
        # print("phrase ->", phrase)
        if phrase not in uniq_words and phrase in uniq_ref:
            if parsed_sent.count(phrase) >= uniq_ref[phrase]:
                uniq_words[phrase] = uniq_ref[phrase]
            else:
                uniq_words[phrase] += 1
    find = sum(uniq_words.values())
    # print(find)
    # print(uniq_words)
    bp = 1 if total >= lref else np.exp(1 - (lref/total))
    # print("bp->", bp)
    # print(find, total)
    return find/parsed_tot


def cumulative_bleu(references, sentence, n):
    """computes the cumulative bleu score"""
    score_hold = []
    bp_hold = []
    lref = min([len(i) for i in references])
    total = len(sentence)
    for i in range(n):
        score_hold.append(ngram_bleu(references, sentence, i + 1))
    # print(score_hold)
    # print(np.average(score_hold))
    bp = 1 if total >= lref else np.exp(1 - (lref/total))
    # print("bp ->", bp)
    return np.average(score_hold) * bp
