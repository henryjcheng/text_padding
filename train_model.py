"""
This module contains code to train model.
The module takes input from model.cfg file.
"""
import random
import pandas as pd
import configparser
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from padding import zero_padding
from net import multilayer_perceptron, CNN

## 0. setting up parameter
config = configparser.ConfigParser()
config.read('model.cfg') 

## PATH
data_path = config['PATH']['data_path']
w2v_path = config['PATH']['w2v_path']

## MODEL_PARAMETERS
model_type = config['MODEL_PARAMETERS']['model_type']
emb_dim = int(config['MODEL_PARAMETERS']['emb_dim'])
pad_method = config['MODEL_PARAMETERS']['pad_method']

## 1. load dataset
df = pd.read_csv(data_path, nrows=5000)

# convert class 4 to class 0
df['Class Index'] = df['Class Index'].replace(4, 0)

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load(w2v_path)
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
max_length = max(df['text_length'])

print(f'max length: {max_length}')

df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim, pad_method))

## 4. load nn architecture
if model_type == 'MP':
    net = multilayer_perceptron()
elif model_type == 'CNN':
    net = CNN()
else:
    raise ValueError(f'\nmodel_type: {model_type} is not recognized.')
    
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

PATH = '../model/cnn/3fc_pad_random.pth'
torch.save(net.state_dict(), PATH)

print('Process complete.')
