from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

dirpath = '/home/roblab15/Documents/FMG_project/data'

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


class Data(Dataset):
    def __init__(self, train=True, dirpath=None):
        # print("ok")
        # dirpath = '/home/roblab15/Documents/FMG_project/data'
        # states dictionary
        filesanddir = [f for f in listdir(dirpath)]
        for i in filesanddir:

            filepath = dirpath + '/' + i
            onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
            if train:
                for j in onlyfiles:
                    # print(join(filepath, j))
                    # print("ok")
                    df = pd.read_csv(join(filepath, j))
                    # print(df['time'])

                    df.drop(['time'], axis=1, inplace=True, errors="ignor")
                    y.append(df['class'])
                    x.append(df.filter(items=['S1', 'S2', 'S3', 'S4']))
            else:
                if i == 'tests':
                    for j in onlyfiles:
                        # print(join(filepath, j))
                        df = pd.read_csv(join(filepath, j))
                        df.drop(['time'], axis=1, inplace=True, errors="ignor")
                        y.append(df['class'])
                        x.append(df.filter(items=['S1', 'S2', 'S3', 'S4']))
                        # train.append(df.filter(items=['S1', 'S2', 'S3', 'S4', 'class']))
                        # x_t.append(df.drop(['class'], axis=1, inplace=False))

                        # print(x_t)

        x_temp1 = pd.concat(x, ignore_index=True)
        y_temp1 = pd.concat(y, ignore_index=True)
        # print("train\n", y_temp1)
        # print("train\n", x_temp1)
        self.X = torch.from_numpy(np.array(x_temp1, dtype=np.float32))
        self.Y = torch.from_numpy(np.array(y_temp1, dtype=np.float32))
        x_temp1 = np.array(x_temp1)
        # print(x_temp1.shape[0])
        self.n_samples = x_temp1.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


# data = Data(train=True, dirpath=dirpath)
# x, y = data[0]
# print(x, y)
