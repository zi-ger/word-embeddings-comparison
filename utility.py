import csv
import glob
import html
import os
import re
import sys

import contractions
from string import punctuation
# from textblob import TextBlob
from tqdm import tqdm

# Removes single quote from punctuation
punct = punctuation.replace("'", "")

def transform_text(raw_str: str) -> str:
    """Transforms a given string, unescaping and
    removing html tags, usernames, punctuation, 
    extra spaces, lowering the text and expanding
    contractions.

    Args:
        raw_str (str): raw string to process

    Returns:
        str: processed string
    """
    
    # Unescape html sentences
    text = html.unescape(raw_str)
    
    # remove any remaining html tag
    text = re.sub(re.compile('<.*?>'), '', text)
    
    # remove links
    text = re.sub(r'(?:\@|https?\://)\S+', '', text)

    # remove contractions
    text = contractions.fix(text)
    
    # lower text
    text = text.lower()
    
    # replace point and hyphen with space to avoid errors in sentences that have no space between them
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')
    
    # remove ponctuation
    text = text.translate(str.maketrans('', '', punctuation))
    
    # remove extra spaces
    text = ' '.join(text.split())

    # correct misspelled words
    # text = str(TextBlob(text).correct()) 
    # function above was misscorrecting slangs and names

    return text


def open_file(path: str, is_csv: bool = False, encoding: str = 'latin-1', verbose: bool = True) -> list:
    """Opens file and returns list of sentences.

    Args:
        path (str): path to file
        csv (bool, optional): if file is in csv format. Defaults to False.
        encoding (str, optional): file encoding, used if file is csv. Defaults to 'latin-1'.
        verbose (bool, optional): opening file verbosity. Defaults to True.

    Returns:
        list: list of sentences from the file
    """
    
    data = []

    try:
        if verbose:
            print('Opening:', path)
        if is_csv:
            with open(path, newline='', encoding=encoding) as f:
                content = csv.reader(f)
                data = list(content)
        else:
            with open(path, 'r') as f:
                data = f.read().splitlines()

    except Exception as er:
        sys.exit('Error opening file: ' + str(er))

    return data 


def opener_util(path:str, is_csv: bool = False, is_dir: bool = False, verbose: bool = True) -> tuple:
    """Deals with file opening options.

    Args:
        path (str): path to file
        is_csv (bool, optional): if file is csv. Defaults to False.
        is_dir (bool, optional): if path is a directory. Defaults to False.
        verbose (bool, optional): changes function verbosity. Defaults to True.

    Returns:
        tuple: positive and negative lists
    """

    input_path = os.path.abspath(path)
    
    # Initialize lists
    positive_data = []
    negative_data = []

    if is_csv:
        # Checks if file exists
        if not os.path.isfile(input_path):
            sys.exit(f"The specified file '{input_path}' does not exist.")
        else:
            # Checks if file ends with .csv
            if not input_path.endswith('.csv'):
                sys.exit(f"The specified file '{input_path}' is not a valid CSV.")
            else:
                data = open_file(input_path, is_csv=True, encoding='latin_1')

                for line in tqdm(data, desc='Processing data'):
                    try:
                        line_polarity = int(line[0])
                    except ValueError as er:
                        if verbose:
                            print('Ignoring line:', er)
                        continue

                    if line_polarity not in [4, 0]:
                        if verbose:
                            print('Ignoring line: invalid polarity value.')
                        continue

                    line_text = line[5] # TODO - change to 1

                    if line_polarity == 0:
                        negative_data.append(line_text)
                    if line_polarity == 4:
                        positive_data.append(line_text)                     

                return positive_data, negative_data

    else:
        # Checks if path provided is a valid directory
        if not os.path.isdir(input_path):
            sys.exit(f"The specified path '{input_path}' is invalid or does not exist.")
        else:

            # Checks if directory option is True
            if is_dir:
                pos_path = input_path + '/pos/'
                neg_path = input_path + '/neg/'

                # Checks if 'pos' dir exists
                if not os.path.isdir(pos_path):
                    sys.exit("Expected directory 'pos' not found.")
                # Checks if 'neg' dir exists
                if not os.path.isdir(neg_path):
                    sys.exit("Expected directory 'neg' not found.")

                # Opens positive path
                for filename in tqdm(glob.glob(pos_path + '*.txt'), desc='Opening ' + pos_path):
                    positive_data.append(open_file(filename, verbose=False)[0])
                
                # Opens negative path
                for filename in tqdm(glob.glob(neg_path + '*.txt'), desc='Opening ' + neg_path):
                    negative_data.append(open_file(filename, verbose=False)[0])

            # Use single files
            else:
                pos_path = input_path + '/pos.txt'
                neg_path = input_path + '/neg.txt'

                # Checks if 'pos.txt' file exists
                if not os.path.isfile(pos_path):
                    sys.exit("Expected file 'pos.txt' not found.")
                # Checks if 'neg.txt' file exists
                if not os.path.isfile(neg_path):
                    sys.exit("Expected file 'neg.txt' not found.")

                # Opens data
                positive_data = open_file(pos_path)
                negative_data = open_file(neg_path)

            return positive_data, negative_data


def save_file(data: list, path: str, filename: str):
    """Saves a list to a text file.

    Args:
        data (list): data that will be saved
        path (str): path to file
        filename (str): file name with extension
    """

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/' + filename, 'w') as f:
        f.write('\n'.join(data))


def split_data(data: list, proportion: float = 0.1) -> tuple:
    """Split a list based on given proportion

    Args:
        data (list): list to split
        proportion (float, optional): proportion used to split. Defaults to 0.1.

    Returns:
        tuple: result lists
    """

    a_prop = int(round(len(data) * (1 - proportion)))
    b_prop = int(round(len(data) * proportion))

    return data[:a_prop], data[-b_prop:]


def define_polarity(corpus: list, polarity: int) -> list:
    """Defines polarity of a given corpus

    Args:
        corpus (list): list of sentences
        polarity (int): sentence polarity. 1 to positive, 0 to negative.

    Returns:
        list: list of sentences, eg [polarity, 'sentence']
    """

    data = []

    for sentence in tqdm(corpus, desc='Defining polarity'):
        data.append([polarity, sentence])
    
    return data


