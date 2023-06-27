from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

dirpath = '/home/roblab15/Documents/FMG project/data'

clf = neighbors.KNeighborsClassifier()
x_train = []
x_test = []
y_train = []
y_test = []
x = []
y = []
count = 0
# states dictionary
filesanddir = [f for f in listdir(dirpath)]
for i in filesanddir:
    filepath = dirpath + '/' + i
    onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]

    for j in onlyfiles:
        print(join(filepath, j))
        df = pd.read_csv(join(filepath, j))
        # print(df['time'])

        df.drop(['time'], axis=1, inplace=True, errors="ignor")
        y.append(df['class'])
        x.append(df.drop(['class'], axis=1, inplace=False))

        # print("x:\n", x)
        # print("y:\n", y)


x_temp = pd.concat(x, ignore_index=True)
y_temp = pd.concat(y, ignore_index=True)
print(y_temp)
print(x_temp)
x1 = np.array(x_temp, dtype=object)
y1 = np.array(y_temp, dtype=object)

# print(x1)
# print(y1)
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)
print("x\n", x_train, "x\n", x_test,"x\n",  y_train, "x\n", y_test)
# clf.fit(x_train, y_train)
# accuracy = clf.score(x_test, y_test)
# print(accuracy)

