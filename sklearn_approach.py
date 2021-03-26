"""Evaluate embeddings using NaiveBayes or SVM."""

import argparse
import os
import random
import sys
import time

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utility import opener_util, transform_text

array_middle = np.full(100, 0.5)

def load_embedding(path: str, embedding_dim: int, num_tokens: int, word_index: dict) -> list:
    """Open previously generated embedding

    Args:
        path (str): path to file
        embedding_dim (int): embedding size
        num_tokens (int): number of tokens from corpus
        word_index (dict): word index from corpus

    Returns:
        list: embedding dict
    """

    embeddings_index = {}

    with open(path) as f:
        if not len(f.readline().split()) < 3:
            f.seek(0)

        for line in tqdm(f, desc='Opening embedding'):
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print(f'Found {len(embeddings_index)} word vectors.')

    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    print(f'Converted {hits} words ({misses} misses).')

    return embedding_matrix


def transform_sentences(corpus: list, corpus_class: int, embedding: list, word_index: dict) -> list:
    """Calculate sentence vectors based on its words"""

    meant_corpus = []

    for sentence in tqdm(corpus, desc=f'Retrieving average embedding values from list {corpus_class}'):
        to_mean = []
        for word in sentence:
            if word in word_index:
                to_mean.append(embedding[word_index[word]])
            else:
                to_mean.append(array_middle)

        meant_corpus.append((np.mean(to_mean, axis=0), corpus_class))

    return meant_corpus


def main():
    """Core script code."""

    # Define argument parser
    parser = argparse.ArgumentParser(
        prog='keras_approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        Generate and analyse a text corpus using Keras TensorFlow.

        The program also can be fed with already generated representations.

        The corpus provided will be cleaned before any processing, using the following steps:
            - unescape and remove html tags
            - remove usernames, punctuation and extra spaces
            - lower the text
            - expand any contractions
        """)

    # Define positional arguments
    parser.add_argument('train', type=str, help='train corpus path')
    parser.add_argument('test', type=str, help='test corpus path')
    parser.add_argument('embedding_alg', type=str, help='embedding algorithm', choices=['word2vec', 'doc2vec', 'fasttext', 'glove', 'keras'])
    parser.add_argument('classifier', type=str, help='classifier algorithm', choices=['naivebayes', 'svm'])
    parser.add_argument('epochs', type=int, help='chosen embedding epochs')

    # Read arguments from the command line
    args = parser.parse_args()

    # Check if path exists
    if not os.path.exists(args.train):
        sys.exit('The specified path or file for train corpus does not exist.')
    if not os.path.exists(args.test):
        sys.exit('The specified path or file for test corpus does not exist.')

    # Get values from arguments
    train_path = args.train
    test_path = args.test
    embedding_alg = args.embedding_alg
    classifier = args.classifier
    epochs = args.epochs

    print('Loading data...')
    train_pos, train_neg = opener_util(train_path)
    test_pos, test_neg = opener_util(test_path)

    # pre process the entire corpus
    train_pos_data = [transform_text(sentence).split() for sentence in tqdm(train_pos, desc='Transforming positive train data')]
    train_neg_data = [transform_text(sentence).split() for sentence in tqdm(train_neg, desc='Transforming negative train data')]
    test_pos_data = [transform_text(sentence).split() for sentence in tqdm(test_pos, desc='Transforming positive test data')]
    test_neg_data = [transform_text(sentence).split() for sentence in tqdm(test_neg, desc='Transforming negative test data')]

    # create vocabulary
    corpus = []
    corpus.extend(train_pos_data)
    corpus.extend(train_neg_data)

    vocabulary = set()
    for word in corpus:
        vocabulary.update(word)

    # get number of tokens
    num_tokens = len(vocabulary)

    # create word/token dictionary
    word_index = dict()
    for index, word in enumerate(vocabulary):
        word_index.update({word: index})

    # load embedding
    embedding = load_embedding(f'models/{embedding_alg}/{embedding_alg}_movie_reviews_embedding_100DIM_{epochs}EP.txt', 100, num_tokens, word_index)

    # normalize embedding using MinMaxScaler
    scaler = MinMaxScaler(copy=False)
    normalized_embedding = scaler.fit_transform(embedding)

    # extract vectors from sentences and mean the values
    train_pos_vectors = transform_sentences(train_pos_data, 1, normalized_embedding, word_index)
    train_neg_vectors = transform_sentences(train_neg_data, 0, normalized_embedding, word_index)

    test_pos_vectors = transform_sentences(test_pos_data, 1, normalized_embedding, word_index)
    test_neg_vectors = transform_sentences(test_neg_data, 0, normalized_embedding, word_index)

    # split into train and test lists
    # pos_train, pos_test = split_data(positive_vectors, 0.25)
    # neg_train, neg_test = split_data(negative_vectors, 0.25)

    # merge train and test data
    train_data = train_pos_vectors + train_neg_vectors
    test_data = test_pos_vectors + test_neg_vectors

    # shuffle lists
    random.shuffle(train_data)
    random.shuffle(test_data)

    # split into vectors and labels
    x_train, y_train = zip(* train_data)
    x_test, y_test = zip(* test_data)

    # define NaiveBayes or SVM classifier model
    if classifier == 'naivebayes':
        model = MultinomialNB()
    else:
        model = svm.SVC()

    # fit model
    start_time = time.time()
    model = model.fit(x_train, y_train)
    print(f'Model fit done in {time.time() - start_time} seconds.')

    # get accuracy score from model
    print('Calculating model score...')
    score = model.score(x_test, y_test)
    score_line = f'\nAccuracy score: { np.round(score * 100, 3) }%'

    # get classification report
    print('Generating classification report...')
    report_line = classification_report(y_test, model.predict(x_test), target_names=['negative', 'positive'], digits=5)

    # write to file
    name = f'{embedding_alg}/{classifier}_{epochs}EP'
    with open(f'results/{name}.txt', 'w') as f:
        f.write(f'Results - {name}:\n')
        f.write(score_line + '\n')
        f.write('\nClassification Report\n\n')
        f.write(report_line)


if __name__ == '__main__':
    main()
