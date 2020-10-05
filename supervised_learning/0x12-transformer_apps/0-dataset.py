class Dataset:
    """Dataset - loads and preps a dataset for machine translation"""
    def __init__(self):
        self.data_train = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)
        self.data_valid = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        self.tokenizer_pt = tfds.features.text.\
            SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in self.data_train),
                target_vocab_size=2**15)

        self.tokenizer_en = tfds.features.text.\
            SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in self.data_train),
                target_vocab_size=2**15)


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
