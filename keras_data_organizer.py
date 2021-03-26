"""Organize data used in Keras Approach module."""

import itertools
import random

import numpy as np
from tensorflow import data
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tqdm import tqdm

from utility import define_polarity, open_file, split_data, transform_text


class KerasDataOrganizer:
    """Keras Data organizer utility class.

    Process and organize data provided by user and provides all
    information needed to use in keras models.

    Args:
        corpus_path (str): path to file
        split_proportion (float): proportion to split data - eg 0.1 represents 10%. Defaults to 0.2.
    """


    def __init__(self, corpus_path: str = None):

        # open and define polarity of train files
        self.pos_train_sentences = define_polarity(open_file(corpus_path + '/train/pos.txt'), 1)
        self.neg_train_sentences = define_polarity(open_file(corpus_path + '/train/neg.txt'), 0)

        # open and define polarity of validation files
        self.pos_val_sentences = define_polarity(open_file(corpus_path + '/validation/pos.txt'), 1)
        self.neg_val_sentences = define_polarity(open_file(corpus_path + '/validation/neg.txt'), 0)

        # open and define polarity of test files
        self.pos_test_sentences = define_polarity(open_file(corpus_path + '/test/pos.txt'), 1)
        self.neg_test_sentences = define_polarity(open_file(corpus_path + '/test/neg.txt'), 0)

        # divide into label and data lists
        self.train_label, self.train_data = split_data_label(self.pos_train_sentences, self.neg_train_sentences)
        self.validation_label, self.validation_data = split_data_label(self.pos_val_sentences, self.neg_val_sentences)
        self.test_label, self.test_data = split_data_label(self.pos_test_sentences, self.neg_test_sentences)

        # update to list of lists
        self.train_data_lists = [[line] for line in tqdm(self.train_data, desc='Updating train data')]
        # self.validation_data_lists = [[line] for line in tqdm(self.validation_data, desc='Updating validation data')]
        # self.test_data_lists = [[line] for line in tqdm(self.test_data, desc='Updating test data')]

        # get the longest sentence
        self.max_length = max([len(s.split()) for s in self.train_data])
        # print('\n\n\n'+ str(self.max_length) +'\n\n\n')

        # initialize the vectorizer
        self.vectorizer = TextVectorization(output_sequence_length=self.max_length,)
        train_ds = data.Dataset.from_tensor_slices(self.train_data_lists).batch(128)
        self.vectorizer.adapt(train_ds)

        self.vocabulary = self.vectorizer.get_vocabulary()

        print('\nData has been processed.')


    def get_word_index(self) -> dict:
        """Returns word index from vectorizer

        Returns:
            dict: vectorizer word index.
        """

        return dict(zip(self.vocabulary, range(len(self.vocabulary))))


    def get_train_corpus(self) -> tuple:
        """Returns training data

        Returns:
            tuple: x_train, y_train
        """

        x_train = self.vectorizer(np.array([s for s in self.train_data])).numpy()
        y_train = np.asarray(self.train_label)

        return x_train, y_train


    def get_validation_corpus(self) -> tuple:
        """Returns validation data

        Returns:
            tuple: x_val, y_val
        """

        x_val = self.vectorizer(np.array([s for s in self.validation_data])).numpy()
        y_val = np.asarray(self.validation_label)

        return x_val, y_val


    def get_test_corpus(self) -> tuple:
        """Returns test data and label

        Returns:
            tuple: test_data, test_label
        """

        test_data = self.vectorizer(np.array([s for s in self.test_data])).numpy()
        test_label = np.asarray(self.test_label)

        return test_data, test_label


    def get_corpus_sentences_splitted(self) -> tuple:
        """Returns train and test lists of splitted sentences

        Returns:
            tuple: train, test
        """

        train = [line.split() for line in self.train_data]
        test = [line.split() for line in self.test_data]

        return train, test


def split_divide(positive_sentences: list, negative_sentences: list, split_proportion: float) -> tuple:
    """Split, shuffle and divide providen corpus into label and data lists

    Returns:
        tuple: train_label, train_data, test_label, test_data
    """

    print('Dividing data...')
    # Split based on given proportion
    train_p, test_p = split_data(positive_sentences, split_proportion)
    train_n, test_n = split_data(negative_sentences, split_proportion)

    # Join negative and positive corpus
    train_corpus = train_p + train_n
    test_corpus = test_p + test_n

    # Shuffle corpus
    random.shuffle(train_corpus)
    random.shuffle(test_corpus)

    # Divide into sentiment/text
    train_label, train_data = zip(*train_corpus)
    test_label, test_data = zip(*test_corpus)

    return train_label, train_data, test_label, test_data


def split_data_label(positive_sentences: list, negative_sentences: list) -> tuple:
    """Alternate join positive and negative corpora and split into label and data lists

    Returns:
        tuple: corpus_label, corpus_data
    """

    corpus = [sentence for sentence in itertools.chain.from_iterable(itertools.zip_longest(positive_sentences, negative_sentences)) if sentence]

    corpus_label, corpus_data = zip(*corpus)

    return corpus_label, corpus_data
