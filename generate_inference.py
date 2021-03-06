"""
This module loops through all dataset to generate inference, 
then export inferece result to dedicated folder
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

from utility import zero_padding, evaluate_accuracy, model_loader, vocab_clean_up, generate_inference
from nets import multilayer_perceptron, CNN, CNN_kim, CNN_deep

time0 = time.time()

## 0. setting up parameter
config = configparser.ConfigParser()
config.read('model.cfg') 

list_dataset = ['ag_news', 'amazon_review_full', 'amazon_review_polarity', 'dbpedia_ontology', 'yelp_review_full', 'yelp_review_polarity']
list_model = ['MP', 'CNN', 'CNN_kim', 'CNN_deep']
list_pad_method = ['bottom', 'bothside', 'random']

for dataset in list_dataset:

    # setting up parameters
    data_path = f'../data/{dataset}/test.csv'
    w2v_path = f'../model/w2v/{dataset}.model'
    model_save_path = f'../model/cnn/{dataset}'
    dataset = f'{dataset}'

    ## 1. load dataset
    if dataset == 'ag_news':
        df = pd.read_csv(data_path)
        df['label'] = df['Class Index'].replace(4, 0)
        df['text_token'] = df['Description'].apply(lambda x: word_tokenize(x))

        df = df.rename(columns={'Description':'text'})

        classes = ('0', '1', '2', '3')

    elif dataset == 'yelp_review_polarity':
        df = pd.read_csv(data_path, names=['label', 'text']).sample(n=5000, random_state=1)
        df['label'] = df['label'].replace(2, 0)
        df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

        classes = ('0', '1')

    elif dataset == 'yelp_review_full':
        df = pd.read_csv(data_path, names=['label', 'text']).sample(n=5000, random_state=1)
        df['label'] = df['label'].replace(5, 0)
        df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

        classes = ('0', '1', '2', '3', '4')

    elif dataset == 'dbpedia_ontology':
        df = pd.read_csv(data_path, names=['label', 'title', 'text'])
        df['label'] = df['label'].replace(14, 0)
        df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13')

    elif dataset == 'amazon_review_polarity':
        df = pd.read_csv(data_path, names=['label', 'title', 'text']).sample(n=50000, random_state=1)
        df['label'] = df['label'].replace(2, 0)
        df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

        classes = ('0', '1')

    elif dataset == 'amazon_review_full':
        df = pd.read_csv(data_path, names=['label', 'title', 'text']).sample(n=50000, random_state=1)
        df['label'] = df['label'].replace(5, 0)
        df['text_token'] = df['text'].apply(lambda x: word_tokenize(x))

        classes = ('0', '1', '2', '3', '4')

    else:
        print(f'Dataset: {dataset} is not recognized.')

    # create testing dataset to append inference result to
    df_inference = df[['text', 'label']]

    w2v = Word2Vec.load(w2v_path)
    df['text_token'] = df['text_token'].apply(lambda x: vocab_clean_up(x, w2v))
    df['text_length'] = df['text_token'].apply(lambda x: len(x))
    df = df[df['text_length'] > 0].reset_index(drop=True)

    df['embedding'] = df['text_token'].apply(lambda x: w2v[x])

    ## 3. zero pad to max length
    if dataset == 'ag_news':
        max_length = 245
    elif dataset == 'yelp_review_polarity':
        max_length = 1200
        df = df[df['text_length'] <= max_length].reset_index(drop=True)     # remove rows with text length > max in train set
    elif dataset == 'yelp_review_full':
        max_length = 1200
        df = df[df['text_length'] <= max_length].reset_index(drop=True)     # remove rows with text length > max in train set
    elif dataset == 'dbpedia_ontology':
        max_length = 413
        df = df[df['text_length'] <= max_length].reset_index(drop=True)     # remove rows with text length > max in train set
    elif dataset == 'amazon_review_polarity':
        max_length = 657
        df = df[df['text_length'] <= max_length].reset_index(drop=True)     # remove rows with text length > max in train set
    elif dataset == 'amazon_review_full':
        max_length = 586
        df = df[df['text_length'] <= max_length].reset_index(drop=True)     # remove rows with text length > max in train set
    else:
        print(f'Dataset: {dataset} is not recognized.')

    print(f'\nDataset: {dataset}')

    for model in list_model:
        for pad_method in list_pad_method:

            model_name = f'{model}_{pad_method}'
            
            ## MODEL_PARAMETERS
            model_type = f'{model}'
            emb_dim = 50
            pad_method = f'{pad_method}'

            df['embedding'] = df['embedding'].apply(lambda x: zero_padding(x, max_length, emb_dim, pad_method))

            tensor_x = torch.tensor(df['embedding'].tolist())
            tensor_y = torch.tensor(df['label'].tolist(), dtype=torch.long)

            data_test= TensorDataset(tensor_x, tensor_y) # create your datset
            loader_test = DataLoader(data_test, batch_size=32, shuffle=False) # create your dataloader

            dataiter = iter(loader_test)
            text, labels = dataiter.next()

            # load model
            net = model_loader(model_type, dataset)
            model_name_temp = model_name + '.pth'
            model_save_path_full = os.path.join(model_save_path, model_name_temp)

            net.load_state_dict(torch.load(model_save_path_full))
            net.eval()

            list_pred = generate_inference(loader_test, net, classes, model_type)

            df_inference = pd.concat([df_inference, pd.DataFrame(list_pred, columns = [f'{model}_{pad_method}'])], axis=1, sort=False)

            print(f'\t{model}_{pad_method}      Time Elapsed: {round(time.time() - time0, 2)}')
    
    file_name = f'{dataset}.csv'
    file_path = f'../output/inference'
    df_inference.to_csv(os.path.join(file_path, file_name), index=False)
