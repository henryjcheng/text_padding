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

We will implement simple 50 x 50 (equals pad_dimension) CNN here.

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
from padding import zero_padding

## 1. load dataset
df = pd.read_csv('../data/ag_news/train.csv')
# convert class 4 to class 0
df['Class Index'] = df['Class Index'].replace(4, 0)
print(df['Class Index'].value_counts())

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load('../model/w2v/ag_news.model')
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
max_length = max(df['text_length'])

print(f'max length: {max_length}')

emb_dim = 50
pad_method = 'bottom'
df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim, pad_method))

## 4. create nn architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (4, 50))        # input channel, output channel, kernel size
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(1, 1))
        self.fc1 = nn.Linear(239 * 1, 120)      # 120 chosen randomly (< input dimension)
        self.fc2 = nn.Linear(120, 50)           # 50 chosen randomly (< 50)
        self.fc3 = nn.Linear(50, 4)             # 4 = number of classes
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 239 * 1)   
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
#tensor_x = tensor_x.unsqueeze(0)    # reshape 1D vecotr to 2D with 1 x ...., where 1 means 1 channel

train_y = df['Class Index'].tolist()
tensor_y = torch.tensor(train_y, dtype=torch.long)
#tensor_y = tensor_y.unsqueeze(0)
set(train_y)

data_train = TensorDataset(tensor_x, tensor_y) # create datset
loader_train = DataLoader(data_train, batch_size=32, shuffle=True) # create dataloader

## 6. train and save model
for epoch in range(20):
    running_loss = 0.0
    print(f'\nepoch {epoch + 1}')
    for i, data in enumerate(loader_train):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.unsqueeze(1)    # reshape by add 1 to num_channel (parameter: batch_size, num_channel, height, width)
        
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

PATH = '../model/cnn/cnn_pad_bottom.pth'
torch.save(net.state_dict(), PATH)

print('Process complete.')
