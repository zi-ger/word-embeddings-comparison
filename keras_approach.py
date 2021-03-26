"""Keras approach to generate and evaluate word embeddings
   applied in sentiment analysis using a CNN."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot, rcParams
from matplotlib.ticker import MaxNLocator
from tensorflow import keras

from keras_data_organizer import KerasDataOrganizer


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
    parser.add_argument('path', type=str, help='corpus path')
    parser.add_argument('size', type=int, help='embedding size')
    parser.add_argument('epochs', type=int, help='training epochs')

    # Define optional argument
    parser.add_argument('-ex', '--external',
        type=str, help='use external embedding', metavar='path')
    parser.add_argument('-t', '--trainable',
        type=bool, help='train the external embedding', metavar='trainable', default=False)
    parser.add_argument('-s', '--save',
        type=str, help='saving name', metavar='save', default='model')

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
    trainable = args.trainable
    saving_name = args.save

    # Load data
    loaded_data = KerasDataOrganizer(path)

    vocabulary = loaded_data.vocabulary
    vectorizer = loaded_data.vectorizer
    word_index = loaded_data.get_word_index()
    num_tokens = len(vocabulary) + 2

    print('\n...\n')

    # Use default keras approach
    if not embedding_path:
        # saving_name = saving_name + '_default'
        print('Using default keras embedding.')
        embedding_layer = keras.layers.Embedding(num_tokens, embedding_dim, name='embedding')

    # Use other embeddings
    else:
        if trainable:
            saving_name = saving_name + '_trainable-true'
        else:
            saving_name = saving_name + '_trainable-false'
        
        print(f"Using external embedding '{embedding_path}' (trainable={trainable}).")
        embedding_matrix = load_embedding(embedding_path, embedding_dim, num_tokens, word_index)

        embedding_layer = keras.layers.Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=trainable,
            name='embedding',
        )

    model = generate_model(embedding_layer)

    evaluate(epochs, model, vectorizer, loaded_data, saving_name)


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

    with open(path) as emb:
        if not len(emb.readline().split()) < 3:
            emb.seek(0)

        for line in emb:
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


def generate_model(embedding_layer):
    """Generates the keras model.

    Args:
        embedding_layer (Embedding): keras embedding layer
    """

    int_sequences_input = keras.Input(shape=(None, ), dtype='int64',)
    embedded_sequences = embedding_layer(int_sequences_input)

    x = keras.layers.Dropout(0.5)(embedded_sequences)
    x = keras.layers.Conv1D(100, 5, padding="same", activation="relu")(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(int_sequences_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def evaluate(epochs: int, model, vectorizer, data, saving_name):
    """Evaluates the embedding and saves results to file.

    Args:
        epochs: (int): number of epochs
        model (Model): keras Model
        vectorizer (TextVectorization): keras vectorizer
        data (DataProcessor): data processor object
    """

    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(verbose=1, monitor='val_loss', mode='min',
            patience=3, restore_best_weights=False),
        keras.callbacks.ModelCheckpoint(verbose=1, monitor='val_accuracy', mode='max',
            filepath=saving_name+'_best_model.h5', save_best_only=True),
    ]

    print('\nFitting model...')
    x_train, y_train = data.get_train_corpus()
    x_val, y_val = data.get_validation_corpus()

    model_history = model.fit(x_train, y_train, batch_size=100,
        epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks, verbose=1)

    print('Evaluating results...')
    test_data, test_label = data.get_test_corpus()

    print("\nModel results:")
    last_results = model.evaluate(test_data, test_label, verbose=1)
    print(last_results)

    # Load best model from checkpoint callback
    best_model = keras.models.load_model(saving_name+'_best_model.h5')

    print("\nBest model results:")
    best_results = best_model.evaluate(test_data, test_label, verbose=1)
    print(best_results)

    results_file = open(saving_name+'_results.txt', 'w+')
    results_file.write('\nEvaluation results using test dataset(best model):')
    results_file.write(f'\n    Loss: {np.round(best_results[0], 4)} - Accuracy: {np.round(best_results[1] * 100, 4)}%\n')

    results_file.write('\nEvaluation results using test dataset(last epoch model):')
    results_file.write(f'\n    Loss: {np.round(last_results[0], 4)} - Accuracy: {np.round(last_results[1] * 100, 4)}%\n')

    # Get all accuracy and loss values into lists
    acc = np.multiply(model_history.history['accuracy'], 100).tolist()
    val_acc = np.multiply(model_history.history['val_accuracy'], 100).tolist()
    loss = model_history.history['loss']
    val_loss =  model_history.history['val_loss']

    # Get index of best epoch and print to file
    max_val_acc_index = val_acc.index(max(val_acc))
    results_file.write(f'\nBest epoch: {max_val_acc_index + 1} ({max_val_acc_index} in history)\n')
    results_file.write('    Train\n')
    results_file.write(f'      Accuracy: {np.round(acc[max_val_acc_index], 4)}% - Loss: {np.round(loss[max_val_acc_index], 4)}\n')
    results_file.write('    Validation\n')
    results_file.write(f'      Accuracy: {np.round(val_acc[max_val_acc_index], 4)}% - Loss: {np.round(val_loss[max_val_acc_index], 4)}\n')

    results_file.write('\n_________________')
    results_file.write('\nTraining history:\n')
    hist_df = pd.DataFrame(model_history.history)
    hist_df.to_csv(results_file, sep='\t')

    results_file.close()

    # Generate model history plot
    fig, (ax1, ax2) = pyplot.subplots(2, sharex=True)
    fig.suptitle(saving_name)

    acc = [None] + acc
    val_acc = [None] + val_acc
    loss = [None] + loss
    val_loss = [None] + val_loss

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Arial'

    ax1.plot(acc, label='treino', marker='.', color='#4986b3')
    ax1.plot(val_acc, label='validação', marker='.', color='#49B378')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax1.yaxis.set_major_formatter('{x:.2f}%')
    ax1.set_ylabel('Acurácia')
    ax1.grid()

    ax2.plot(loss, label='treino', marker='.', color='#4986b3')
    ax2.plot(val_loss, label='validação', marker='.', color='#49B378')
    ax2.yaxis.set_major_formatter('{x:.2f}')

    ax2.set_ylabel('Perda')
    ax2.set_xlabel('Épocas')
    x_ticks = list(range(1, len(loss)))
    ax2.set_xticklabels(x_ticks)

    ax2.grid()

    pyplot.savefig(saving_name + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
