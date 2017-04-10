import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train_vectors_path = './train_vectors.txt'
test_vectors_path = './test_vectors.txt'

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
print(model.predict_proba([[1, 2, 3, 4]]))
