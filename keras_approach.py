import argparse
import time
import sys
import os

import numpy as np
from tensorflow import keras
from tqdm import tqdm

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec

from keras_data_organizer import KerasDataOrganizer


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
    parser.add_argument('size', type=int, help='embedding size')
    parser.add_argument('epochs', type=int, help='training epochs')

    # Define optional argument
    parser.add_argument('-ex', '--external', type=str, help='use external embedding', metavar='path')

    # Read arguments from the command line
    args = parser.parse_args()

    # Check if path exists
    if not os.path.exists(args.path):
        sys.exit('The specified path or file does not exist.')

    # Get values from arguments
    embedding_dim = args.size
    embedding_path = args.external
    epochs = args.epochs
    path = args.path

    print('Loading data...')
    loaded_data = KerasDataOrganizer(path)

    vocabulary = loaded_data.vocabulary
    vectorizer = loaded_data.vectorizer
    word_index = loaded_data.get_word_index()

    num_tokens = len(vocabulary) + 2

    print('\n...\n')

    # Use default keras approach
    if not embedding_path:
        embedding_layer = keras.layers.Embedding(num_tokens, embedding_dim)
    
    # Use other embeddings 
    else:
        embedding_matrix = load_embedding(embedding_path, embedding_dim, num_tokens, word_index)

        embedding_layer = keras.layers.Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )

    evaluate(num_tokens, epochs, embedding_layer, vectorizer, loaded_data)


def load_embedding(path: str, embedding_dim: int, num_tokens: int, word_index: dict) -> list:
    """Open previously generated embedding

    Args:
        path (str): path to file
        embedding_dim (int): embedding size
        num_tokens (int): number of tokens from corpus
        word_index (dict): word index from corpus

    Returns:
        list: embedding matrix array
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


def evaluate(num_tokens: int, epochs: int, embedding_layer, vectorizer, data):
    """Evaluates the embedding and saves results to file.

    Args:
        num_tokens (int): [description]
        embedding_layer (Embedding): keras embedding layer
        vectorizer (TextVectorization): keras vectorizer
        data (DataProcessor): data processor object
    """

    int_sequences_input = keras.Input(shape=(None,), dtype='int64',)
    embedded_sequences = embedding_layer(int_sequences_input)
    x = keras.layers.Conv1D(100, 5, activation='relu', padding='same')(embedded_sequences)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Conv1D(100, 5, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Conv1D(100, 5, activation='relu', padding='same')(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(int_sequences_input, preds)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    input('Press Enter to continue...')

    print('Fitting model...')
    x_train, y_train, x_val, y_val = data.get_training_corpus()
    model.fit(x_train, y_train, batch_size=100, epochs=epochs, validation_data=(x_val, y_val))

    print('Evaluating results...')
    test_data, test_label = data.get_test_corpus()
    results = model.evaluate(test_data, test_label, verbose=2)
    print(results)

    while True:
        print()
        test_input = [input('sentence: ').lower()]

        if test_input[0] == 'exit.':
            sys.exit()

        prediction = model.predict(vectorizer(np.array(test_input)).numpy(), verbose=1)

        print(f'Prediction: {np.round(prediction[0][0]* 100, 2)}%.')


if __name__ == '__main__':
    main()
