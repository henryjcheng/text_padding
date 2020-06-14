__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This module contains functions to trains word2vec model from text contained in a pandas DataFrame.

Need to input 2 command-line arguments:
    1. path of data, including file name
    2. path to model save directory

Future development:
    1. create config file and load arguments from config

For conveniency, routinely used paths are identified here:
/media/henry/data/School/master_thesis/data/ag_news/train.csv 
/media/henry/data/School/master_thesis/model/w2v/ag_news
"""
import os
import sys
import time
import pandas as pd
import multiprocessing
import configparser
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
## first time uinsg nltk, uncomment the following 2 lines
# import nltk
# nltk.download('punkt')

def train_w2v(df, emb_dim, min_count):
    """
    This function is the main function that trains word2vec model from given text.
    The column containing text needs to have column name as 'text', eg. df['text']

    df: pandas DataFrame comtaining at least one column named 'text'
    emb_dim: embedding dimension, dimension of the word2vec model
    min_count: minimum frequency count of word in the word2vec model
    """
    print('Start training word2vec...')
    time0 = time.time()
    
    # tokenize
    print('\tTokenization...')
    df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

    # train model
    print('\tTrain word2vec model...')
    print(f'\t\t- dimension: {emb_dim}')
    print(f'\t\t- min count: {min_count}')
    w2v = Word2Vec(df['text_token'].tolist(),
                   size=emb_dim,
                   window=5,
                   min_count=min_count,
                   negative=15,
                   iter=10,
                   workers=multiprocessing.cpu_count())
    
    time_diff = round(time.time() - time0, 2)
    print(f'Training complete.    Time elapsed: {time_diff}')

    return w2v



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('w2v.cfg')

    # setting up parameters
    data_path = config['PATH']['data_path']
    model_save_path = config['PATH']['model_save_path']

    dataset = config['MODEL']['dataset']
    model_name = config['MODEL']['model_name']
    emb_dim = int(config['MODEL']['embedding_dimension'])
    min_freq = int(config['MODEL']['min_frequency'])

    # preprocessing
    df = pd.read_csv(data_path)
    if dataset == 'ag_news':
        df_text = df[['Description']].reset_index(drop=True).rename(columns={'Description':'text'})
    else:
        print(f'Dataset: {dataset} not recognized.')
    
    # train model
    w2v = train_w2v(df_text, emb_dim, min_freq)

    # save trained model
    w2v.save(os.path.join(model_save_path, model_name))
