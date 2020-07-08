"""
This module contains code to evaluate model against test set.
The module takes input from model.cfg file.
"""
import os
import random
import pandas as pd
import configparser
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utility import zero_padding, evaluate_accuracy, model_loader, vocab_clean_up
from nets import multilayer_perceptron, CNN, CNN_kim, CNN_deep

## 0. setting up parameter
config = configparser.ConfigParser()
config.read('model.cfg') 

## PATH
data_path = config['PATH']['test_data_path']
w2v_path = config['PATH']['w2v_path']
model_save_path = config['PATH']['model_save_path']
model_name = config['PATH']['model_name']
dataset = config['PATH']['dataset']

## MODEL_PARAMETERS
model_type = config['MODEL_PARAMETERS']['model_type']
emb_dim = int(config['MODEL_PARAMETERS']['emb_dim'])
pad_method = config['MODEL_PARAMETERS']['pad_method']

classes = ('0', '1', '2', '3')
## 1. load dataset
if dataset == 'ag_news':
    df = pd.read_csv(data_path)
    df['label'] = df['Class Index'].replace(4, 0)
    df = df.rename(columns={'Class Index':'label'})
    df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))
elif dataset == 'yelp_review_polarity':
    df = pd.read_csv(data_path, names=['label', 'text'])
    df['label'] = df['label'].replace(2, 0)
    df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))
else:
    print(f'Dataset: {dataset} is not recognized.')

w2v = Word2Vec.load(w2v_path)
df['text_token'] = df['text_token'].apply(lambda x: vocab_clean_up(x, w2v))
df['text_length'] = df['text_token'].apply(lambda x: len(x))
df = df[df['text_length'] > 0].reset_index(drop=True)

df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
if dataset == 'ag_news':
    max_length = 245
elif dataset == 'yelp_review_polarity':
    max_length = 1200
    df = df[df['text_length'] <= max_length].reset_index(drop=True)     # remove rows with text length > max in train set
else:
    print(f'Dataset: {dataset} is not recognized.')

print(f'max length: {max_length}')

df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim, pad_method))

tensor_x = torch.tensor(df['embedding'].tolist())
tensor_y = torch.tensor(df['label'].tolist(), dtype=torch.long)

data_test= TensorDataset(tensor_x, tensor_y) # create your datset
loader_test = DataLoader(data_test, batch_size=32, shuffle=False) # create your dataloader

dataiter = iter(loader_test)
text, labels = dataiter.next()

# load model
net = model_loader(model_type, dataset)
model_name_temp = model_name + '.pth'
model_save_path_full = os.path.join(model_save_path, model_name_temp)

net.load_state_dict(torch.load(model_save_path_full))
net.eval()

evaluate_accuracy(loader_test, net, classes, model_type)
        