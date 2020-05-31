__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This module contains functions to trains word2vec model from text contained in a pandas DataFrame.

Need to input 2 command-line arguments:
    1. path of data, including file name
    2. path to model save directory

For conveniency, routinely used paths are identified here:
/media/henry/data/School/master_thesis/data/ag_news/train.csv
"""
import sys
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
## first time uinsg nltk, uncomment the following 2 lines
# import nltk
# nltk.download('punkt')

def train_w2v():
    """
    This function is the main function that trains word2vec model from given text
    """

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    df_text = df[['Description']].reset_index(drop=True).rename(columns={'Description':'text'})

    print(df_text.head())