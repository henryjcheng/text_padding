"""
This module contains code to evaluate model against test set.
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
data_path = config['PATH']['test_data_path']
w2v_path = config['PATH']['w2v_path']
model_save_path = config['PATH']['model_save_path']

## MODEL_PARAMETERS
model_type = config['MODEL_PARAMETERS']['model_type']
emb_dim = int(config['MODEL_PARAMETERS']['emb_dim'])
pad_method = config['MODEL_PARAMETERS']['pad_method']

## 1. load dataset
df = pd.read_csv(data_path)
# convert class 4 to class 0
df['Class Index'] = df['Class Index'].replace(4, 0)
print(df['Class Index'].value_counts())

## 2. apply tokenization and embedding
df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

w2v = Word2Vec.load(w2v_path)
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
if model_type == 'MP':
    net = multilayer_perceptron()
elif model_type == 'CNN':
    net = CNN()
else:
    raise ValueError(f'\nmodel_type: {model_type} is not recognized.')
net.load_state_dict(torch.load(model_save_path))

correct = 0
total = 0
with torch.no_grad():
    for data in loader_test:
        text, labels = data
        if model_type == 'CNN':
            text = text.unsqueeze(1)    # reshape text to add 1 channel

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
        if model_type == 'CNN':
            text = text.unsqueeze(1)    # reshape text to add 1 channel
            
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