"""
This module contains neural network classes
"""
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
