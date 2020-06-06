__author__ = "Henry Cheng"
__email__ = "henryjcheng@gmail.com"
__status__ = "dev"
"""
This module contains codes to train CNN multi-class classification model.
Program flow:
    1. load data as pd.DataFrame
    2. apply tokenization and embedding
    3. zero pad to max length
    4. create nn architecture
    5. create training pipeline
    6. train and save model

We will start with simple 3 FC layers and focus on getting pipeline running before expanding 
our architecture.

Future dev:
    1. refactor preprocessing code to one function
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

## 1. load dataset
df = pd.read_csv('../data/ag_news/train.csv')
# convert class 4 to class 0
df['Class Index'] = df['Class Index'].replace(4, 0)

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load('../model/w2v/ag_news.model')
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
max_length = max(df['text_length'])

print(f'max length: {max_length}')

emb_dim = 50
df['embedding'] = df['embedding'].apply(lambda x: zero_padding_bothside(x, max_length, emb_dim))

## 4. create nn architecture
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

net = Net()
    
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f'\ndevice: {device}')

net.to(device)

## 5. create training pipeline
train_x = df['embedding'].tolist()
tensor_x = torch.tensor(train_x)

train_y = df['Class Index'].tolist()
tensor_y = torch.tensor(train_y, dtype=torch.long)
set(train_y)

data_train = TensorDataset(tensor_x, tensor_y) # create your datset
loader_train = DataLoader(data_train, batch_size=32, shuffle=True) # create your dataloader

## 6. train and save model
for epoch in range(5):
    running_loss = 0.0
    print(f'\nepoch {epoch + 1}')
    for i, data in enumerate(loader_train):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i and i % 200 == 0:
            print(f'\tbatch {i}    loss: {running_loss/200}')
        running_loss = 0.0

PATH = '../model/cnn/3fc_pad_bothside.pth'
torch.save(net.state_dict(), PATH)

print('Process complete.')
