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
import paramaters
from clasifiers import transforms

# dirpath = paramaters.parameters.dirpath

x_train = []
x_test = []
y_train = []
y_test = []
x = []
y = []
featurs = []
x_t = []
y_t = []
y_real_train = []
x_real_train = []
count = 0
sample_rate = 10


# items = paramaters.parameters.items


class Data(Dataset):
    def __init__(self, args_config, train=True, dirpath=None, items=None):
        cut = 200
        # states dictionary
        filesanddir = [f for f in listdir(dirpath)]

        df_mean, df_mean_test = data_agmuntation.find_mean(filesanddir, dirpath, items)
        self.n_classes = 4

        if train:
            for dir_name in filesanddir:
                filepath = dirpath + '/' + dir_name
                onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
                count = 0
                if not dir_name == 'tests':
                    for file_name in onlyfiles:

                        df = pd.read_csv(join(filepath, file_name))
                        # cuts first 100 samples
                        df = df.iloc[cut:, :].reset_index(drop=True)
                        df.drop(['time'], axis=1, inplace=True, errors="ignor")

                        num_location = file_name[file_name.find('_') + 1]

                        if len(file_name) > 13:
                            num_location = file_name[file_name.find('_') + 2]
                            temp = int(num_location)
                            mul = file_name[file_name.find('_') + 1]
                            temp += int(mul) * 10
                            num_location = str(temp)
                        # if dir_name == 'relaxed':
                        for index in items:
                            df[index] -= df_mean.loc[index, num_location]  # subtracts the bias val from the original
                        y_train.append(df['class'])
                        x_train.append(df.filter(items=items))


        else:
            for dir_name in filesanddir:
                count = 0
                filepath = dirpath + '/' + dir_name
                onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
                if dir_name == 'tests':

                    for file_name in onlyfiles:

                        df_test = pd.read_csv(join(filepath, file_name))
                        df_test = df_test.iloc[cut:, :].reset_index(drop=True)
                        df_test.drop(['time'], axis=1, inplace=True, errors="ignor")
                        num_location_test = file_name[file_name.find('_') + 1]
                        count += 1
                        if len(file_name) > 13:
                            temp = int(num_location_test)
                            mul = file_name[file_name.find('_') + 1]
                            temp += int(mul) * 10
                            num_location_test = str(temp)
                        for index in items:
                            df_test[index] -= df_mean_test.loc[
                                index, num_location_test]  # subtracts the mean val from the original

                        y_test.append(df_test['class'])
                        x_test.append(df_test.filter(items=items))

        if train:
            featurs = pd.concat(x_train, ignore_index=True)
            labels = pd.concat(y_train, ignore_index=True)

            # for i in range(labels_train.shape[0]):
            #     if labels_train[i] == 3 or labels_train[i] == 2:
            #         labels_train[i] = 1
            #         self.n_classes = 2
            # full_data = pd.merge(featurs_train, labels_train, left_index=True, right_index=True)
        else:
            featurs = pd.concat(x_test, ignore_index=True)
            labels = pd.concat(y_test, ignore_index=True)

            # for i in range(labels_test.shape[0]):
            #     if labels_test[i] == 3 or labels_test[i] == 2:
            #         labels_test[i] = 1
            #         self.n_classes = 2

            # full_data = pd.merge(featurs_test, labels_test, left_index=True, right_index=True)

        # print(full_data)

        # data agmuntations
        # normalization
        # corrent_featurs = data_agmuntation.min_max_norm(corrent_featurs)
        corrent_featurs = data_agmuntation.stdnorm(featurs)

        # rooling mean of 10 points
        corrent_featurs = data_agmuntation.filter(corrent_featurs, args_config=args_config)
        corrent_labels = labels

        # print(full_data)
        # true_featurs = mean_filter_df[items]
        # true_labels = mean_filter_df['class']
        # corrent_labels = true_labels
        #
        # corrent_featurs = true_featurs

        # if train:
        # # data_agmuntation
        #     corrent_featurs, corrent_labels = data_agmuntation.scaling(true_featurs, corrent_featurs, corrent_labels, true_labels)
        #     corrent_featurs, corrent_labels = data_agmuntation.flip(true_featurs, corrent_featurs, corrent_labels, true_labels)
        #     corrent_featurs, corrent_labels = data_agmuntation.rotation(true_featurs, corrent_featurs, corrent_labels,  true_labels)
        # corrent_featurs, corrent_labels = data_agmuntation.permutation(true_featurs, corrent_featurs , corrent_labels,  true_labels)
        # corrent_featurs, corrent_labels = data_agmuntation.magnitude_wrap(true_featurs, corrent_featurs, corrent_labels,true_labels)

        # corrent_featurs = transforms.lda_transform(corrent_featurs, corrent_labels)

        # corrent_featurs = transforms.PCA_transform(corrent_featurs, corrent_labels)

        self.X = torch.from_numpy(np.array(corrent_featurs, dtype=np.float32))
        if train:
            # self.Y = torch.from_numpy(np.array(corrent_labels['class'], dtype=np.float32))
            self.Y = torch.from_numpy(np.array(corrent_labels, dtype=np.float32))
        else:
            self.Y = torch.from_numpy(np.array(corrent_labels, dtype=np.float32))

        x_temp1 = np.array(corrent_featurs)
        self.n_samples = x_temp1.shape[0]
        self.n_featurs = x_temp1.shape[1]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples

#
# data = Data(train=True, dirpath=dirpath, items=items)
# x, y = data[0]
# print(x, y)
