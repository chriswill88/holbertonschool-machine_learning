#!/usr/bin/env python3
"""this module is created for task 3"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset - loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
        self.data_train, train_info = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            shuffle_files=True,
            split='train', as_supervised=True, with_info=True)
        self.data_valid, valid_info = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            shuffle_files=True,
            split='validation', as_supervised=True, with_info=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.ml = max_len
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(self.filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000).padded_batch(
            batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(
            self.filter_max_length).padded_batch(batch_size)

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

        pt_i.extend(self.tokenizer_pt.encode(pt.numpy()[:]))
        en_i.extend(self.tokenizer_en.encode(en.numpy()[:]))
        pt_i.append(ptt+1)
        en_i.append(ent+1)
        return pt_i, en_i

    def tf_encode(self, pt, en):
        """wrapper for encode instance method"""
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    def filter_max_length(self, x, y):
        """filters the length"""
        max_length = self.ml
        return tf.logical_and(
            tf.size(x) <= max_length, tf.size(y) <= max_length)
