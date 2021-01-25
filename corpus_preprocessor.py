import argparse
import glob
import os
import sys

from tqdm import tqdm

from utility import opener_util
from utility import save_file
from utility import transform_text


def main():
    # Define argument parser
    parser = argparse.ArgumentParser(
        prog='corpus_preprocessor', 
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description="""
        Open and process a sentence corpus, removing noises, usernames, html tags, 
        punctuations, expanding contractions and correcting misspelled words, 
        saves processed corpus to file.

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

    # Define positional argument
    parser.add_argument('path', type=str, help='input corpus path')

    # Define exclusive optional arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--csv', help='use csv file', action='store_true')
    group.add_argument('-d', '--dir', help='use directory with multiple files', action='store_true')

    # Read arguments from the command line
    args = parser.parse_args()

    # Get arguments from parser
    path = args.path
    is_csv = args.csv
    is_dir = args.dir

    # Open files
    positive_data, negative_data = opener_util(path, is_csv, is_dir)

    # Transform data
    positive_data = [transform_text(line) for line in tqdm(positive_data, desc='Transforming positive data')]
    negative_data = [transform_text(line) for line in tqdm(negative_data, desc='Transforming negative data')]

    # Get absolute path
    abs_path = os.path.abspath(path)

    # Get get dir name if csv
    if is_csv:
        abs_path = os.path.dirname(abs_path)

    # Save files to input path
    save_file(positive_data, abs_path + '/processed', 'pos.txt')
    save_file(negative_data, abs_path + '/processed', 'neg.txt')


if __name__ == '__main__':
    main()