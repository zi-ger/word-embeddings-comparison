import argparse
import time
import sys
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec

from tqdm import tqdm

from utility import opener_util


def main():
    # Define argument parser
    parser = argparse.ArgumentParser(
        prog='embedding_generator',
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description="""
        Generate word embeddings using Word2Vec, Doc2Vec and fastText algorithms.
        
        The corpus provided will be cleaned before any processing, using the following steps:
            - unescape and remove html tags
            - remove usernames, punctuation and extra spaces
            - lower the text
            - expand any contractions

        * By default the program will look for two text files into the directory provided:
            - pos.txt
            - neg.txt

            These files must contain one sentence per line.
        
        ** If --csv option is used, attempt to the following:
            - Only one file is expected

            - Column 0 : sentence label, must be: 
                0: if negative
                4: if positive
            
            - Column 1 : sentence text
        
        *** If --dir option is used, attempt to the following:
            - The program will look for two separate folders 'pos/' and 'neg/'
            
            - It will open all '.txt' files in each folder

            - One file equals one line in the new processed output file 
        """)

    # Define positional arguments
    parser.add_argument('path', type=str, help='corpus path')
    parser.add_argument('size', type=int, help='embedding size', default=100)
    parser.add_argument('epochs', type=int, help='training epochs', default=10)
    parser.add_argument('model', type=str, help='algorithm model', choices=['word2vec', 'doc2vec', 'fasttext'])

    # Define not exclusive optional arguments
    parser.add_argument('--save', type=str, help='filename to save generated embeddings', metavar='filename')

    # Defines exclusive optional arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--csv', help='use csv file', action='store_true')
    group.add_argument('-d', '--dir', help='use directory with multiple files', action='store_true')

    # Read arguments from the command line
    args = parser.parse_args()

    # Check if path exists
    if not os.path.exists(args.path):
        sys.exit('The specified path or file does not exist.')

    # Get values from arguments
    save_filename = args.save
    embedding_dim = args.size
    epochs = args.epochs
    model = args.model
    is_csv = args.csv
    is_dir = args.dir
    path = args.path

    positive_data, negative_data = opener_util(path, is_csv, is_dir)

    corpus = positive_data + negative_data
    corpus = [line.split() for line in corpus]

    # Check arguments
    if model == 'word2vec':
        print('Generating Word2Vec embedding...')
        generate_w2v_embedding(corpus, embedding_dim, epochs, save_filename)
    
    elif model == 'doc2vec':
        print('Generating Doc2Vec embedding...')
        generate_d2v_embedding(corpus, embedding_dim, epochs, save_filename)

    elif model == 'fasttext':
        print('Generating fastText embedding...')
        generate_fasttext_embedding(corpus, embedding_dim, epochs, save_filename)


def generate_w2v_embedding(corpus: list, embedding_dim: int, epochs: int, save_filename: str):
    """Generate Word2Vec Embedding and saves to file"""

    start_time = time.time()
    w2v_model = Word2Vec(size=embedding_dim, min_count=3, iter=50, workers=8)
    print('Building vocab...')
    w2v_model.build_vocab(corpus)
    print('Training model...')
    w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=epochs)
    print(f'Done in {time.time() - start_time} seconds.')
    print('Saving model...')
    w2v_model.wv.save_word2vec_format(f'word2vec_{save_filename}_embedding_{embedding_dim}DIM_{epochs}EP.txt', binary=False)


def generate_d2v_embedding(corpus: list, embedding_dim: int, epochs: int, save_filename: str):
    """Generate Doc2Vec Embedding and saves to file"""

    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(corpus)]

    start_time = time.time()
    d2v_model = Doc2Vec(vector_size=embedding_dim, min_count=3, epochs=epochs,)
    print('Building vocab...')
    d2v_model.build_vocab(tagged_data)
    print('Training model...')
    d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
    print(f'Done in {time.time() - start_time} seconds.')
    print('Saving model...')
    
    d2v_model.wv.save_word2vec_format(f'doc2vec_{save_filename}_embedding_{embedding_dim}DIM_{epochs}EP.txt', binary=False)


def generate_fasttext_embedding(corpus: list, embedding_dim: int, epochs: int, save_filename: str):
    """Generate fastText Embedding and saves to file"""

    start_time = time.time()
    ft_model = FastText(size=embedding_dim, min_count=3, iter=50, workers=8)
    print('Building vocab...')
    ft_model.build_vocab(corpus)
    print('Training model...')
    ft_model.train(corpus, total_examples=ft_model.corpus_count, total_words=ft_model.corpus_total_words, epochs=epochs)
    print(f'Done in {time.time() - start_time} seconds.')
    print('Saving model...')
    ft_model.wv.save_word2vec_format(f'fasttext_{save_filename}_embedding_{embedding_dim}DIM_{epochs}EP.txt', binary=False)


if __name__ == '__main__':
    main()
