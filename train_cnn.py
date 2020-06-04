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
    5. create nn architecture
    6. create training pipeline
    7. train and save model
    8. evaluate model performance

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

## 1. load dataset
df = pd.read_csv('../data/ag_news/train.csv')

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load('../model/w2v/ag_news.model')
df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

## 3. zero pad to max length
df['text_length'] = df['text_token'].apply(lambda x: len(x))
#max_length = max(df['text_length'])
max_length = 245

print(f'max length: {max_length}')

emb_dim = 50
df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim))

## 4. convert to pytorch tensor
list_to_append = []

for array in df['embedding']:
    list_to_append.append(torch.tensor(array))

## 5. create nn architecture
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

## 6. create training pipeline
train_x = df['embedding'].tolist()
tensor_x = torch.tensor(train_x)

train_y = df['Class Index'].tolist()
tensor_y = torch.tensor(train_y, dtype=torch.long)
set(train_y)

my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True) # create your dataloader

## 7. train and save model
for epoch in range(5):
    running_loss = 0.0
    print(f'\nepoch {epoch + 1}')
    for i, data in enumerate(my_dataloader):
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

PATH = '../model/cnn/fc3.pth'
torch.save(net.state_dict(), PATH)

print('Process complete.')
