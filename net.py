"""
This module contains neural network classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class multilayer_perceptron(nn.Module):
    def __init__(self):
        super(multilayer_perceptron, self).__init__()
        self.fc1 = nn.Linear(245 * 50 * 1, 120) # 120 chosen randomly (< 245*50*1)
        self.fc2 = nn.Linear(120, 50)           # 50 chosen randomly (< 50)
        self.fc3 = nn.Linear(50, 4)             # 4 = number of classes
    
    def forward(self, x):
        x = x.view(-1, 245 * 50 * 1)   # token length, w2v embedding dimension, channel
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

class CNN_kim(nn.Module):
    def __init__(self):
        super(CNN_kim, self).__init__()
        self.conv1_a = nn.Conv2d(1, 1, (3, 50))    # channel 1 of conv, with kernel=3 
        self.conv1_b = nn.Conv2d(1, 1, (4, 50))    # channel 2 of conv, with kernel=4
        self.conv1_c = nn.Conv2d(1, 1, (5, 50))    # channel 3 of conv, with kernel=5
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(1, 1))
        self.pool_b = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1))
        self.pool_c = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))
        self.fc1 = nn.Linear(240 * 3, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 4)
    
    def forward(self, x):
        x0 = x
        x = self.pool(F.relu(self.conv1_a(x)))
        y = self.pool_b(F.relu(self.conv1_b(x0)))
        z = self.pool_c(F.relu(self.conv1_c(x0)))
        x = x.view(-1, 240 * 1)
        y = y.view(-1, 240 * 1)
        z = z.view(-1, 240 * 1)
        x = torch.cat((x, y, z), dim=1)    # combine results from three conv
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

