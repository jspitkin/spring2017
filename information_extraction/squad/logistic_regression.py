import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from random import randint
from random import shuffle

train_vectors_path = './train_vectors.txt'
test_vectors_path = './test_vectors.txt'

def train():
    # Read in the training feature vectors
    feature_vectors = []
    labels = []
    with open(train_vectors_path, 'r') as dataset_file:
        for line in dataset_file:
            labels.append(line.split()[0])
            feature_vectors.append(line.split()[1:])
    X = np.array(feature_vectors)
    y = np.array(labels)

    # Train a logistic regression model with the training examples
    model = LogisticRegression()
    model.fit(X, y)
    return model

def train_even():
    # Read in the training feature vectors
    feature_vectors = []
    labels = []
    with open(train_vectors_path, 'r') as dataset_file:
        for line in dataset_file:
            labels.append(line.split()[0])
            feature_vectors.append(line.split()[1:])
    pos_count = 0
    balanced_count_vectors = []
    balanced_count_labels = []
    for index,label in enumerate(labels):
        if label == '1':
            pos_count += 1
            balanced_count_vectors.append(feature_vectors[index])
            balanced_count_labels.append(labels[index])

    neg_samples = 0
    used_indices = set([])
    while(neg_samples < pos_count):
        random_index = randint(0, len(feature_vectors) - 1)
        if random_index not in used_indices and labels[random_index] == '0':
            balanced_count_vectors.append(feature_vectors[random_index])
            balanced_count_labels.append(labels[random_index])
            used_indices.add(random_index)
            neg_samples += 1
    shuffled_indices = [i for i in range(0, neg_samples + pos_count)] 
    shuffle(shuffled_indices)

    train_vectors = []
    train_labels = []
    for index in shuffled_indices:
        train_vectors.append(balanced_count_vectors[index])
        train_labels.append(balanced_count_labels[index])
    X = np.array(train_vectors)
    y = np.array(train_labels)

    print('# of examples:', len(train_vectors))
    print('# of positive:', pos_count)
    print('# of negative:', neg_samples)

    model = LogisticRegression()
    model.fit(X, y)
    return model
