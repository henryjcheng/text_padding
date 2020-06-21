"""
This module contains code to train model.
The module takes input from model.cfg file.
"""
import os
import time
import random
import pandas as pd
import configparser
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utility import zero_padding, model_loader
from net import multilayer_perceptron, CNN, CNN_kim, CNN_deep

## 0. setting up parameter
config = configparser.ConfigParser()
config.read('model.cfg') 

## PATH
data_path = config['PATH']['train_data_path']
w2v_path = config['PATH']['w2v_path']
model_save_path = config['PATH']['model_save_path']
model_name = config['PATH']['model_name']

## MODEL_PARAMETERS
sample = config['MODEL_PARAMETERS'].getboolean('sample')
model_type = config['MODEL_PARAMETERS']['model_type']
emb_dim = int(config['MODEL_PARAMETERS']['emb_dim'])
pad_method = config['MODEL_PARAMETERS']['pad_method']

batch_size = int(config['MODEL_PARAMETERS']['batch_size'])
shuffle = config['MODEL_PARAMETERS'].getboolean('shuffle')
epoch = int(config['MODEL_PARAMETERS']['epoch'])

## 1. load dataset
if sample:
    df = pd.read_csv(data_path, nrows=5000)
else:
    df = pd.read_csv(data_path)

# convert class 4 to class 0
df['Class Index'] = df['Class Index'].replace(4, 0)

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load(w2v_path)
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
if sample:
    max_length = 245
else:
    max_length = max(df['text_length'])

print(f'sample is {sample},    training size: {df.shape[0]},    max length: {max_length}')

df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim, pad_method))

## 4. load nn architecture
net = model_loader(model_type)
    
# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f'\ndevice: {device}')

net.to(device)

## 5. create training pipeline
tensor_x = torch.tensor(df['embedding'].tolist())
tensor_y = torch.tensor(df['Class Index'].tolist(), dtype=torch.long)

data_train = TensorDataset(tensor_x, tensor_y) # create your datset
loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle) # create your dataloader

## 6. train and save model
for run in range(epoch):
    running_loss = 0.0
    print(f'\nepoch {run + 1}')
    time0_epoch = time.time()

    for i, data in enumerate(loader_train):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        if model_type != 'MP':
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
    
    time_diff_epoch = round(time.time() - time0_epoch, 2)
    print(f'\tTime elapsed: {time_diff_epoch}')


    model_name_temp = model_name + f'_epoch{run+1}' + '.pth'
    model_save_path_full = os.path.join(model_save_path, model_name_temp)
    torch.save(net.state_dict(), model_save_path_full)

model_name_temp = model_name + '.pth'
model_save_path_full = os.path.join(model_save_path, model_name_temp)
torch.save(net.state_dict(), model_save_path_full)

print('\nProcess complete.')
