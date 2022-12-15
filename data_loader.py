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
sample_rate = 10
items = ['B1', 'B2', 'S1', 'S2', 'S3', 'S4']


class Data(Dataset):
    def __init__(self, train=True, dirpath=None, emd=False):
        # states dictionary
        filesanddir = [f for f in listdir(dirpath)]

        for dir_name in filesanddir:
            filepath = dirpath + '/' + dir_name
            onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
            if dir_name == 'relaxed':
                df_mean = pd.DataFrame(columns=onlyfiles)
                c = 0
                for file_name in onlyfiles:
                    df = pd.read_csv(join(filepath, file_name))
                    df = df.iloc[100:, :]
                    df.drop(['time'], axis=1, inplace=True, errors="ignor")
                    df = df.filter(items=items)
                    df_mean[file_name] = df.mean()
                    # print(df_mean.loc['S1', j])

        for dir_name in filesanddir:
            filepath = dirpath + '/' + dir_name
            onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]

            if train:

                for file_name in onlyfiles:
                    # print(join(filepath, j))
                    # print("ok")
                    df = pd.read_csv(join(filepath, file_name))
                    # print(df['time'])
                    df = df.iloc[100:, :]
                    y.append(df['class'])
                    df.drop(['time'], axis=1, inplace=True, errors="ignor")
                    if dir_name == 'relaxed':
                        for index in items:
                            df[index] -= df_mean.loc[index, file_name]
                    x.append(df.filter(items=items))

            else:
                if dir_name == 'tests':
                    for file_name in onlyfiles:
                        # print(join(filepath, file_name))
                        df = pd.read_csv(join(filepath, file_name))
                        df.drop(['time'], axis=1, inplace=True, errors="ignor")
                        y.append(df['class'])
                        x.append(df.filter(items=items))

        x_temp1 = pd.concat(x, ignore_index=True)
        y_temp1 = pd.concat(y, ignore_index=True)
        self.X = torch.from_numpy(np.array(x_temp1, dtype=np.float32))
        self.Y = torch.from_numpy(np.array(y_temp1, dtype=np.float32))
        x_temp1 = np.array(x_temp1)
        self.n_samples = x_temp1.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples

#
# data = Data(train=True, dirpath=dirpath)
# x, y = data[0]
# print(x, y)
