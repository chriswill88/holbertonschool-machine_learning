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

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

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
        pt_i = self.tokenizer_pt.encode(pt.numpy())
        en_i = self.tokenizer_en.encode(en.numpy())

        return pt_i, en_i
