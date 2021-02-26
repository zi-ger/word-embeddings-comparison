import argparse
import random
import time
import sys
import os

import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from utility import opener_util
from utility import transform_text
from utility import split_data


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
    meant_corpus = []
    
    for sentence in tqdm(corpus, desc=f'Retrieving average embedding values from list {corpus_class}'):
        to_mean = []
        for word in sentence:
            to_mean.append(embedding[word_index[word]])

        meant_corpus.append((np.mean(to_mean, axis=0), corpus_class))

    return meant_corpus


def main():
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
    parser.add_argument('path', type=str, help='corpus path')
    # parser.add_argument('size', type=int, help='embedding size')
    # parser.add_argument('epochs', type=int, help='training epochs')

    # Read arguments from the command line
    args = parser.parse_args()

    # Check if path exists
    if not os.path.exists(args.path):
        sys.exit('The specified path or file does not exist.')

    # Get values from arguments
    # embedding_dim = args.size
    # epochs = args.epochs
    path = args.path

    print('Loading data...')
    pos, neg = opener_util(path)

    # pre process the entire corpus
    positive_data = [transform_text(sentence).split() for sentence in tqdm(pos, desc='Transforming positive data')]
    negative_data = [transform_text(sentence).split() for sentence in tqdm(neg, desc='Transforming negative data')]
    
    # create vocabulary
    vocabulary = set()
    for word in positive_data + negative_data:
        vocabulary.update(word)

    # get number of tokens
    num_tokens = len(vocabulary)

    # create word/token dictionary
    word_index = dict()
    for index, word in enumerate(vocabulary):
        word_index.update({word: index})

    # load embedding
    # embedding = load_embedding('data/models/word2vec/word2vec_movie_review_embedding_100DIM_25EP.txt', 100, num_tokens, word_index)
    # embedding = load_embedding('data/models/doc2vec/doc2vec_movie_review_embedding_100DIM_25EP.txt', 100, num_tokens, word_index)
    # embedding = load_embedding('data/models/fasttext/fasttext_movie_review_embedding_100DIM_25EP.txt', 100, num_tokens, word_index)
    
    embedding = load_embedding('data/models/glove/glove.6B.100d.txt', 100, num_tokens, word_index)

    # normalize embedding using MinMaxScaler
    scaler = MinMaxScaler(copy=False)
    normalized_embedding = scaler.fit_transform(embedding)

    # extract vectors from sentences and mean the values
    positive_vectors = transform_sentences(positive_data, 1, normalized_embedding, word_index)
    negative_vectors = transform_sentences(negative_data, 0, normalized_embedding, word_index)

    # split into train and test lists
    pos_train, pos_test = split_data(positive_vectors, 0.25)
    neg_train, neg_test = split_data(negative_vectors, 0.25)
 
    # merge train and test data
    train_data = pos_train + neg_train
    test_data = pos_test + neg_test
    
    # shuffle lists
    random.shuffle(train_data)
    random.shuffle(test_data)

    # split into vectors and labels
    x_train, y_train = zip(* train_data)
    x_test, y_test = zip(* test_data)

    # define NaiveBayesClassifier or SVM model
    model = MultinomialNB()
    # model = svm.SVC()

    # fit model
    start_time = time.time()
    model = model.fit(x_train, y_train)
    print(f'Model fit done in {time.time() - start_time} seconds.')
    
    # get accuracy score from model
    score = model.score(x_test, y_test)
    score_line = f'\nAccuracy score: { np.round(score * 100, 3) }%'

    # get classification report
    report_line = classification_report(y_test, model.predict(x_test), target_names=['negative', 'positive'], digits=5)

    # write to file
    name = 'result'
    with open(f'results/{ name }.txt', 'w') as f: 
        f.write(f'Results - {name}:\n')
        f.write(score_line + '\n')
        f.write('\nClassification Report\n\n')
        f.write(report_line)


if __name__ == '__main__':
    main()
