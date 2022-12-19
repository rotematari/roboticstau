from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data_agmuntation

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
    def __init__(self, train=True, dirpath=None, items=None):
        # states dictionary
        filesanddir = [f for f in listdir(dirpath)]
        df_mean = data_agmuntation.find_mean(filesanddir, dirpath, items)

        for dir_name in filesanddir:
            filepath = dirpath + '/' + dir_name
            onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]

            if train:
                if not dir_name == 'tests':
                    for file_name in onlyfiles:

                        df = pd.read_csv(join(filepath, file_name))
                        # cuts first 100 samples
                        df = df.iloc[100:, :].reset_index(drop=True)
                        y.append(df['class'])
                        df.drop(['time'], axis=1, inplace=True, errors="ignor")
                        num_location = file_name[file_name.find('_') + 1]

                        # if dir_name == 'relaxed':
                        for index in items:
                            df[index] -= df_mean.loc[index, num_location]  # subtracts the mean val from the original
                        x.append(df.filter(items=items))

            else:
                if dir_name == 'tests':
                    for file_name in onlyfiles:
                        # print(join(filepath, file_name))
                        df = pd.read_csv(join(filepath, file_name))
                        df.drop(['time'], axis=1, inplace=True, errors="ignor")
                        y.append(df['class'])
                        x.append(df.filter(items=items))

        featurs = pd.concat(x, ignore_index=True)
        labels = pd.concat(y, ignore_index=True)
        full_data = pd.merge(featurs, labels, left_index=True, right_index=True)
        # print(full_data)

# data agmuntations
        # rooling mean of 10 points
        mean_filter_df = data_agmuntation.filter(full_data)

        # addig the mean data to the full data
        full_data = pd.concat([full_data, mean_filter_df], ignore_index=True)

        # print(full_data)
        featurs = full_data[items]
        labels = full_data['class']

        # featurs = data_agmuntation.min_max_norm(featurs)
        featurs = data_agmuntation.stdnorm(featurs)
        self.X = torch.from_numpy(np.array(featurs, dtype=np.float32))
        self.Y = torch.from_numpy(np.array(labels, dtype=np.float32))
        x_temp1 = np.array(featurs)
        self.n_samples = x_temp1.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


#
data = Data(train=True, dirpath=dirpath, items=items)
# x, y = data[0]
# print(x, y)
