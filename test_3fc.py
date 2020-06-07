__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This module contains codes to test/evaluate CNN multi-class classification model.
Program flow:

Future dev:
    1. combine this into train_3fc.py
"""
import random
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

def zero_padding_random(list_to_pad, max_length, pad_dimension):
    """
    This function takes a list and add list of zeros until max_length is reached.
    The number of zeroes in added list is determined by pad_dimension, which is the 
    same as the dimension of the word2vec model.

    Padding is done randomly, ie padding verctors are inserted into text randomly.
    1. randomly generate a list of index for padding vectors
    2. fill an empty numpy array with dimension = max_length

    This function is intended to handle one list only so it can be passed 
    into a dataframe as a lambda function.
    """
    # find number of padding vector needed
    num_pad = max_length - len(list_to_pad)

    # vector_pad = np.zeros(pad_dimension)
    vector_pad = np.asarray([0] * pad_dimension, dtype=np.float32)
    vector_pad = [vector_pad]    # convert to list of np.ndarray so we can append together 

    position_random = random.sample(range(0, max_length-1), num_pad)
    index_list_to_pad = 0
    list_temp = np.empty((0, pad_dimension), dtype=np.float32)
    for position in range(max_length):
        if position in position_random:
            vector_to_append = vector_pad
        else:
            vector_to_append = [list_to_pad[index_list_to_pad]]
            index_list_to_pad += 1

        list_temp = np.append(list_temp, vector_to_append, axis=0)

    return list_temp

def zero_padding_bothside(list_to_pad, max_length, pad_dimension):
    """
    This function takes a list and add list of zeros until max_length is reached.
    The number of zeroes in added list is determined by pad_dimension, which is the 
    same as the dimension of the word2vec model.

    Padding is done to both side of the text. When required number of padding vector is 
    odd, the extra vector is add to the right (bottom) side.  

    This function is intended to handle one list only so it can be passed 
    into a dataframe as a lambda function.
    """
    # find number of padding vector needed
    num_pad = max_length - len(list_to_pad)

    # vector_pad = np.zeros(pad_dimension)
    vector_pad = np.asarray([0] * pad_dimension, dtype=np.float32)
    vector_pad = [vector_pad]    # convert to list of np.ndarray so we can append together 

    num_each_side = int(num_pad/2)
    iteration = 0
    list_each_side = np.empty((0, pad_dimension), dtype=np.float32)
    while iteration < num_each_side:
        list_each_side = np.append(list_each_side, vector_pad, axis=0)
        iteration += 1

    list_to_pad = np.append(list_each_side, list_to_pad, axis=0)
    list_to_pad = np.append(list_to_pad, list_each_side, axis=0)

    # add one more pad to the right side when odd number of padding vector
    if num_pad%2 == 1:
        list_to_pad = np.append(list_to_pad, vector_pad, axis=0)
    
    return list_to_pad

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
df['embedding'] = df['embedding'].apply(lambda x: zero_padding_random(x, max_length, emb_dim))

test_x = df['embedding'].tolist()
tensor_x = torch.tensor(test_x)

test_y = df['Class Index'].tolist()
tensor_y = torch.tensor(test_y, dtype=torch.long)

data_test= TensorDataset(tensor_x, tensor_y) # create your datset
loader_test = DataLoader(data_test, batch_size=32, shuffle=True) # create your dataloader

dataiter = iter(loader_test)
text, labels = dataiter.next()

# load model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(245 * 50 * 1, 120) # 120 chosen randomly (< 245*50*1)
        self.fc2 = nn.Linear(120, 50)           # 50 chosen randomly (< 50)
        self.fc3 = nn.Linear(50, 4)             # 4 = number of classes
    
    def forward(self, x):
        x = x.view(-1, 245 * 50 * 1)   # token length, w2v embedding dimension, channel
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

PATH = '../model/cnn/3fc_pad_random.pth'
net = Net()
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