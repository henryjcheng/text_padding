__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This module contains codes to train CNN multi-class classification model.
Program flow:
    1. load data as pd.DataFrame
    2. apply tokenization and embedding
    3. zero pad to max length
    4. convert to pytorch tensor
    5. run through nn architecture

We will start with simple 3 FC layers and focus on getting pipeline running before expanding 
our architecture.
"""
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F

def zero_padding(list_to_pad, max_length, pad_dimension):
    """
    This function takes a list and add list of zeros until max_length is reached.
    The number of zeroes in added list is determined by pad_dimension, which is the 
    same as the dimension of the word2vec model.
    This function is intended to handle one list only so it can be passed 
    into a dataframe as a lambda function.
    """
    # find number of padding vector needed
    num_pad = max_length - len(list_to_pad)

    # vector_pad = np.zeros(pad_dimension)
    vector_pad = np.asarray([0] * pad_dimension, dtype=np.float32)
    vector_pad = [vector_pad]    # convert to list of np.ndarray so we can append together 

    iteration = 0
    while iteration < num_pad:
        list_to_pad = np.append(list_to_pad, vector_pad, axis=0)
        iteration += 1
    
    return list_to_pad

## 1. load dataset
df = pd.read_csv('../data/ag_news/train.csv')

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load('../model/w2v/ag_news.model')
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
max_length = max(df['text_length'])

print(f'max length: {max_length}')

emb_dim = 50
df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim))

test = df['embedding'].to_numpy(dtype="float32")
print(test)

## 4. convert to pytorch tensor
train = torch.from_numpy(test)

print(train)