from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd
from sklearn import neighbors

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


dirpath = '/home/roblab15/Documents/FMG_project/data'

clf = neighbors.KNeighborsClassifier()
x_train = []
x_test = []
y_train = []
y_test = []
x = []
y = []
x_t = []
y_t = []
y_real_train = []
x_real_train = []
count = 0
# states dictionary
filesanddir = [f for f in listdir(dirpath)]
for i in filesanddir:

    filepath = dirpath + '/' + i
    onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    if i == 'tests':
        for j in onlyfiles:
            print(join(filepath, j))
            df = pd.read_csv(join(filepath, j))
            df.drop(['time'], axis=1, inplace=True, errors="ignor")
            y_t.append(df['class'])
            x_t.append(df.filter(items=['S1', 'S2', 'S3', 'S4']))

            # x_t.append(df.drop(['class'], axis=1, inplace=False))

            # print(x_t)
    else:
        for j in onlyfiles:
            print(join(filepath, j))
            df = pd.read_csv(join(filepath, j))
            # print(df['time'])

            df.drop(['time'], axis=1, inplace=True, errors="ignor")
            y.append(df['class'])
            x.append(df.filter(items=['S1', 'S2', 'S3', 'S4']))

            # print("x:\n", x)
            # print("y:\n", y)

x_temp1 = pd.concat(x, ignore_index=True)
y_temp1 = pd.concat(y, ignore_index=True)
print("train\n", y_temp1)
print("train\n", x_temp1)
X_train = np.array(x_temp1)
Y_train = np.array(y_temp1)

x_temp2 = pd.concat(x_t, ignore_index=True)
y_temp2 = pd.concat(y_t, ignore_index=True)
print("test\n", y_temp2)
print("test\n", x_temp2)
X_test = np.array(x_temp2)
Y_test = np.array(y_temp2)

noRm = 1
if noRm:
    # Normalize with mean and std
    scaler = StandardScaler()
    scaler.fit(X_train)
    X = scaler.transform(X_train)  # X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)
    x_mean = scaler.mean_
    x_std = scaler.scale_
else:
    # Normalize with min and max
    x_max = np.max(X_train, 0)
    x_min = np.min(X_train, 0)
    X = (X_train - x_min) / (x_max - x_min)

inputs, labels = shuffle_in_unison(X_train, Y_train)
inputs_test, labels_test = shuffle_in_unison(X_test, Y_test)
# print(inputs)
# print(X)
# X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.00001, random_state=42)

# print(X_test)
# print(X_train)
# print(y_test)
names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM',  # 'Gaussian Process',
         'Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost']

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=20),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier()]

# iterate over classifiers
scores = []
for name, clf in zip(names, classifiers):
    clf.fit(list(inputs), list(labels))

    score = clf.score(inputs_test, labels_test)  # Evaluate on test data
    scores.append(score)
    print(name, score)
scores = dict(zip(names, scores))
