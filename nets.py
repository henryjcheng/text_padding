"""
This module contains neural network classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class multilayer_perceptron(nn.Module):
    def __init__(self, dataset):
        if dataset == 'ag_news':
            self.fc1_in = 245 * 50 * 1   
            self.fc1_out = 120           # same as fc2_in
            self.fc2_out = 50            # same as fc3_in
            self.fc3_out = 4             # same as number of classes
        elif dataset == 'yelp_review_polarity':
            self.fc1_in = 1200 * 50 * 1   
            self.fc1_out = 120           # same as fc2_in
            self.fc2_out = 50            # same as fc3_in
            self.fc3_out = 2             # same as number of classes
        elif dataset == 'yelp_review_full':
            self.fc1_in = 1200 * 50 * 1   
            self.fc1_out = 120           # same as fc2_in
            self.fc2_out = 50            # same as fc3_in
            self.fc3_out = 5             # same as number of classes
        elif dataset == 'dbpedia_ontology':
            self.fc1_in = 413 * 50 * 1   
            self.fc1_out = 120           # same as fc2_in
            self.fc2_out = 50            # same as fc3_in
            self.fc3_out = 14            # same as number of classes
        elif dataset == 'amazon_review_polarity':
            self.fc1_in = 657 * 50 * 1   
            self.fc1_out = 120           # same as fc2_in
            self.fc2_out = 50            # same as fc3_in
            self.fc3_out = 2             # same as number of classes
        else:
            raise ValueError(f'Dataset: {dataset} not recognized.')

        super(multilayer_perceptron, self).__init__()
        self.fc1 = nn.Linear(self.fc1_in, self.fc1_out)   
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)         
        self.fc3 = nn.Linear(self.fc2_out, self.fc3_out)           
    
    def forward(self, x):
        x = x.view(-1, self.fc1_in)   # token length, w2v embedding dimension, channel
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, dataset):
        if dataset == 'ag_news':
            self.fc1_in = 239 * 1
            self.n_class = 4
        elif dataset == 'yelp_review_polarity':
            self.fc1_in = 1194 * 1
            self.n_class = 2
        elif dataset == 'yelp_review_full':
            self.fc1_in = 1194 * 1
            self.n_class = 5
        elif dataset == 'dbpedia_ontology':
            self.fc1_in = 407 * 1
            self.n_class = 14
        elif dataset == 'amazon_review_polarity':
            self.fc1_in = 651 * 1
            self.n_class = 2
        else:
            raise ValueError(f'Dataset: {dataset} not recognized.')

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (4, 50))        # input channel, output channel, kernel size
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(1, 1))
        self.fc1 = nn.Linear(self.fc1_in, 120)      # 120 chosen randomly (< input dimension)
        self.fc2 = nn.Linear(120, 50)           # 50 chosen randomly (< 50)
        self.fc3 = nn.Linear(50, self.n_class)             # 4 = number of classes
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.fc1_in)   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_kim(nn.Module):
    def __init__(self, dataset):
        if dataset == 'ag_news':
            self.fc1_in = 240
            self.n_class = 4
        elif dataset == 'yelp_review_polarity':
            self.fc1_in = 1195
            self.n_class = 2
        elif dataset == 'yelp_review_full':
            self.fc1_in = 1195
            self.n_class = 5
        elif dataset == 'dbpedia_ontology':
            self.fc1_in = 408
            self.n_class = 14
        elif dataset == 'amazon_review_polarity':
            self.fc1_in = 652
            self.n_class = 2
        else:
            raise ValueError(f'Dataset: {dataset} not recognized.')

        super(CNN_kim, self).__init__()
        self.conv1_a = nn.Conv2d(1, 1, (3, 50))    # channel 1 of conv, with kernel=3 
        self.conv1_b = nn.Conv2d(1, 1, (4, 50))    # channel 2 of conv, with kernel=4
        self.conv1_c = nn.Conv2d(1, 1, (5, 50))    # channel 3 of conv, with kernel=5
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(1, 1))
        self.pool_b = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1))
        self.pool_c = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))
        self.fc1 = nn.Linear(self.fc1_in * 3, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, self.n_class)
    
    def forward(self, x):
        x0 = x
        x = self.pool(F.relu(self.conv1_a(x)))
        y = self.pool_b(F.relu(self.conv1_b(x0)))
        z = self.pool_c(F.relu(self.conv1_c(x0)))
        x = x.view(-1, self.fc1_in * 1)
        y = y.view(-1, self.fc1_in * 1)
        z = z.view(-1, self.fc1_in * 1)
        x = torch.cat((x, y, z), dim=1)    # combine results from three conv
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_deep(nn.Module):
    def __init__(self, dataset):
        if dataset == 'ag_news':
            self.kernel2 = 120
            self.kernel3 = 59
            self.kernel4 = 28
            self.fc1_in = 27
            self.n_class = 4
        elif dataset == 'yelp_review_polarity':
            self.kernel2 = 598
            self.kernel3 = 297
            self.kernel4 = 147
            self.fc1_in = 147
            self.n_class = 2
        elif dataset == 'yelp_review_full':
            self.kernel2 = 598
            self.kernel3 = 297
            self.kernel4 = 147
            self.fc1_in = 147
            self.n_class = 5
        elif dataset == 'dbpedia_ontology':
            self.kernel2 = 205
            self.kernel3 = 101
            self.kernel4 = 49
            self.fc1_in = 47
            self.n_class = 14
        elif dataset == 'yelp_review_polarity':
            self.kernel2 = 327
            self.kernel3 = 163
            self.kernel4 = 81
            self.fc1_in = 79
            self.n_class = 2
        else:
            raise ValueError(f'Dataset: {dataset} not recognized.')

        super(CNN_deep, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (3, 50))

        # Conv block 1
        self.conv2 = nn.Conv2d(64, 64, (3, 1))
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(self.kernel2, 1), stride=(1, 1))  # halve the dimension

        # Conv block 2
        self.conv3 = nn.Conv2d(128, 128, (3, 1))
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(self.kernel3, 1), stride=(1, 1))    # halve the dimension

        # Conv block 3
        self.conv4 = nn.Conv2d(256, 256, (3, 1))
        self.conv4_bn = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(self.kernel4, 1), stride=(1, 1))   # halve the dimension

        self.fc1 = nn.Linear(self.fc1_in * 1 * 512, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, self.n_class)
    
    def forward(self, x):
        x = self.conv1(x)

        # Conv block 1
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        x = torch.cat((x, x), dim=1)        # doubling the feature space

        # Conv block 2
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool3(x)
        x = torch.cat((x, x), dim=1)        # doubling the feature space

        # Conv block 3
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.pool4(x)
        x = torch.cat((x, x), dim=1)        # doubling the feature space

        x = x.view(-1, self.fc1_in * 1 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x