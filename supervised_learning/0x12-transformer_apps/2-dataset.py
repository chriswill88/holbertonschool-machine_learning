#!/usr/bin/env python3
"""this module is created for task 1"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset - loads and preps a dataset for machine translation"""
    def __init__(self):
        self.data_train = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)

        self.data_valid = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        tokenize_dataset - creates subword tokenizers for our dataset
        @data - tf.data.Dataset formated (pt, en)
        return tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encode - encodes a translation into token
        @pt: tf.Tensor portuguese sentence
        @en: tf.Tensor english sentence
        return pt_tokens, en_tokens
        """
        ptt = self.tokenizer_pt.vocab_size
        ent = self.tokenizer_en.vocab_size

        pt_i = [ptt]
        en_i = [ent]
        pt_i.extend(self.tokenizer_pt.encode(pt.numpy()))
        en_i.extend(self.tokenizer_en.encode(en.numpy()))

        pt_i.append(ptt+1)
        en_i.append(ent+1)
        return pt_i, en_i

    def tf_encode(self, pt, en):
        """
            tf_encode -
                that acts as a tensorflow wrapper
                for the encode instance method
        """
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
