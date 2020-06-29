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

## MODEL_PARAMETERS
model_type = config['MODEL_PARAMETERS']['model_type']
emb_dim = int(config['MODEL_PARAMETERS']['emb_dim'])
pad_method = config['MODEL_PARAMETERS']['pad_method']

classes = ('0', '1', '2', '3')

## 1. load dataset
df = pd.read_csv(data_path)
# convert class 4 to class 0
df['Class Index'] = df['Class Index'].replace(4, 0)
print(df['Class Index'].value_counts())

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load(w2v_path)
df['text_token'] = df['text_token'].apply(lambda x: vocab_clean_up(x, w2v))
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
#max_length = max(df['text_length'])
max_length = 245    # specify max length from train set

print(f'max length: {max_length}')

df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim, pad_method))

tensor_x = torch.tensor(df['embedding'].tolist())
tensor_y = torch.tensor(df['Class Index'].tolist(), dtype=torch.long)

data_test= TensorDataset(tensor_x, tensor_y) # create your datset
loader_test = DataLoader(data_test, batch_size=32, shuffle=True) # create your dataloader

dataiter = iter(loader_test)
text, labels = dataiter.next()

# load model
net = model_loader(model_type)
model_name_temp = model_name + '.pth'
model_save_path_full = os.path.join(model_save_path, model_name_temp)

net.load_state_dict(torch.load(model_save_path_full))
net.eval()

evaluate_accuracy(loader_test, net, classes, model_type)
        