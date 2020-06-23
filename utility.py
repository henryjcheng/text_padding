"""
This module contains misc. function used in train_model.py
"""
import random
import numpy as np
import torch
import nets

def zero_padding(list_to_pad, max_length, pad_dimension, pad_method='bottom'):
    """
    This function takes a list and add list of zeros until max_length is reached.
    The number of zeroes in added list is determined by pad_dimension, which is the 
    same as the dimension of the word2vec model.

    There are three modes available:
        bottom - zero vectors are added to the bottom/right side of the embedding
        bothside - zero vectors are added to both side of the embedding
        random - zero vectors's positions are randomly inserted into the embedding

    This function is intended to handle one list only so it can be passed 
    into a dataframe as a lambda function.
    """
    # find number of padding vector needed
    num_pad = max_length - len(list_to_pad)

    # create zero vector based on pad_dimension
    vector_pad = np.asarray([0] * pad_dimension, dtype=np.float32)
    vector_pad = [vector_pad]    # convert to list of np.ndarray so we can append together 

    if pad_method == 'bottom':
        iteration = 0
        while iteration < num_pad:
            list_to_pad = np.append(list_to_pad, vector_pad, axis=0)
            iteration += 1

    elif pad_method == 'bothside':
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

    elif pad_method == 'random':
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
        list_to_pad = list_temp

    else:
        raise ValueError(f'{pad_method} is not a valid padding method.')

    return list_to_pad

def evaluate_accuracy(loader_test, net, classes, model_type):
    """
    This function takes pytorch data loader, pytorch class for NN, 
    and a tuple of class labels to calculate accuracy at macro and class levels
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader_test:
            text, labels = data
            if model_type != 'MP':
                text = text.unsqueeze(1)    # reshape text to add 1 channel

            outputs = net(text)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nAccuracy: {100 * correct/total}%')

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for batch, data in enumerate(loader_test):
            text, labels = data
            if model_type != 'MP':
                text = text.unsqueeze(1)    # reshape text to add 1 channel

            outputs = net(text)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(4):
        print('Accuracy of class %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / (class_total[i] + .000001)))

def model_loader(model_type):
    """
    This function loads model from net.py
    """
    if model_type == 'MP':
        net = nets.multilayer_perceptron()
    elif model_type == 'CNN':
        net = nets.CNN()
    elif model_type == 'CNN_kim':
        net = nets.CNN_kim()
    elif model_type == 'CNN_deep':
        net = nets.CNN_deep()
    else:
        raise ValueError(f'\nmodel_type: {model_type} is not recognized.')

    return net



if __name__ == "__main__":
    pass
