import re
import sys
import random
import numpy as np

from tqdm import tqdm
from string import punctuation

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import data

from utility import define_polarity
from utility import open_file
from utility import split_data
from utility import transform_text


class KerasDataOrganizer:
    """Keras Data organizer utility class.

    Process and organize data provided by user and provides all information needed to use in keras models.

    Args:
        corpus_path (str): path to file
        split_proportion (float): percentage proportion to split data - eg 0.1 to represent 10%. Defaults to 0.2.
    """


    def __init__(self, corpus_path: str = None, split_proportion: float = 0.20):
        self.split_proportion = split_proportion

        self.positive_sentences = open_file(corpus_path + '/pos.txt')
        self.negative_sentences = open_file(corpus_path + '/neg.txt')

        self.positive_sentences = define_polarity(self.positive_sentences, 1)
        self.negative_sentences = define_polarity(self.negative_sentences, 0)

        self.train_label, self.train_data, self.test_label, self.test_data = split_divide(self.positive_sentences, self.negative_sentences, split_proportion)

        self.train_data_lists = [[line] for line in tqdm(self.train_data, desc='Updating train data')]
        self.test_data_lists = [[line] for line in tqdm(self.test_data, desc='Updating test data')]

        self.max_length = max([len(s.split()) for s in self.train_data + self.test_data])

        self.vectorizer = TextVectorization(output_sequence_length=self.max_length,)
        text_ds = data.Dataset.from_tensor_slices(self.train_data_lists + self.test_data_lists).batch(128)
        self.vectorizer.adapt(text_ds)

        self.vocabulary = self.vectorizer.get_vocabulary()

        print('\nData has been processed.')


    def get_word_index(self) -> dict:
        """Returns word index from vectorizer 
        
        Returns:
            dict: vectorizer word index.
        """
        
        return dict(zip(self.vocabulary, range(len(self.vocabulary))))


    def get_training_corpus(self) -> tuple:
        """Returns training and validation data 
        
        Returns:
            tuple: x_train, y_train, x_val and y_val
        """
        
        a_train, a_val = split_data(self.train_data, 0.15)
        b_train, b_val = split_data(self.train_label, 0.15)

        x_train = self.vectorizer(np.array([s for s in a_train])).numpy()
        y_train = np.asarray(b_train)
        
        x_val = self.vectorizer(np.array([s for s in a_val])).numpy()
        y_val = np.asarray(b_val)

        return x_train, y_train, x_val, y_val


    def get_test_corpus(self) -> tuple:
        """Returns test data and label 
        
        Returns:
            tuple: test_data, test_label
        """
        
        test_data = self.vectorizer(np.array([s for s in self.test_data])).numpy()
        test_label = np.asarray(self.test_label)
        
        return test_data, test_label


    def get_unprocessed_corpus(self) -> tuple:
        """Returns positive and negative corpus combined without any processing 
        
        Returns:
            tuple: sentences, labels
        """
        
        labels, sentences = zip(* self.positive_sentences + self.negative_sentences)
        
        return sentences, labels


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
