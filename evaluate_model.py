"""
This module contains code to evaluate model against test set.
The module takes input from model.cfg file.
"""
import random
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from padding import zero_padding
from net import multilayer_perceptron

## 1. load dataset
df = pd.read_csv('../data/ag_news/test.csv')
# convert class 4 to class 0
df['Class Index'] = df['Class Index'].replace(4, 0)
print(df['Class Index'].value_counts())

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load('../model/w2v/ag_news.model')
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
#max_length = max(df['text_length'])
max_length = 245    # specify max length from train set

print(f'max length: {max_length}')

emb_dim = 50
pad_method = 'bottom'
df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim, pad_method))

test_x = df['embedding'].tolist()
tensor_x = torch.tensor(test_x)

test_y = df['Class Index'].tolist()
tensor_y = torch.tensor(test_y, dtype=torch.long)

data_test= TensorDataset(tensor_x, tensor_y) # create your datset
loader_test = DataLoader(data_test, batch_size=32, shuffle=True) # create your dataloader

dataiter = iter(loader_test)
text, labels = dataiter.next()

# load model
PATH = '../model/cnn/test.pth'
net = multilayer_perceptron()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in loader_test:
        text, labels = data
        outputs = net(text)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\nAccuracy: {100 * correct/total}%')

class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
with torch.no_grad():
    for batch, data in enumerate(loader_test):
        text, labels = data
        outputs = net(text)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

classes = ('0', '1', '2', '3')

for i in range(4):
    print('Accuracy of class %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / (class_total[i] + .000001)))