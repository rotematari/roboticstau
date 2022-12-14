from os import listdir
from os.path import join, isfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import paramaters

from learn import agmontation
items = paramaters.parameters.items


# finds the mean of relaxed state
df_mean = pd.DataFrame()


def find_mean(states_dir, data_dir_path, items):
    for dir_name in states_dir:
        filepath = data_dir_path + '/' + dir_name
        onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        # clac mean val for relaxed state in each data acquisition session

        if dir_name == 'relaxed':
            # df_mean = pd.DataFrame()
            c = 0
            for file_name in onlyfiles:
                df = pd.read_csv(join(filepath, file_name))
                df = df.iloc[100:, :].reset_index(drop=True)
                df.drop(['time'], axis=1, inplace=True, errors="ignor")
                df = df.filter(items=items)
                num = file_name[file_name.find('_') + 1]
                df_mean[num] = df.mean()  # new data frame of mean vals
        if dir_name == 'tests':
            filepath = data_dir_path + '/' + dir_name
            onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
            count_relaxed = 0
            for file_name in onlyfiles:
                if 'relaxed' in file_name:
                    count_relaxed += 1
            df_mean_test = pd.DataFrame()
            for file_name in onlyfiles:
                if 'relaxed' in file_name:
                    df_test = pd.read_csv(join(filepath, file_name))
                    df_test = df_test.iloc[100:, :].reset_index(drop=True)
                    df_test.drop(['time'], axis=1, inplace=True, errors="ignor")
                    df_test = df_test.filter(items=items)
                    num = file_name[file_name.find('_') + 1]
                    df_mean_test[num] = df_test.mean()

            return df_mean, df_mean_test


# 1 filter the data by 10 point avg
def filter(feature_df):

    feature_df_roll = feature_df.rolling(window=10).mean()

    feature_df_roll = feature_df_roll.dropna().reset_index(drop=True)

    return feature_df_roll


# min max normalization
def min_max_norm(true_featurs):

    true_featurs = pd.DataFrame(true_featurs, columns=items)

    df_norm = (true_featurs - true_featurs.min()) / (true_featurs.max() - true_featurs.min())

    return df_norm


# standardization

def stdnorm(feature_df):

    true_featurs = pd.DataFrame(feature_df, columns=items)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(feature_df)
    X = scaler.transform(feature_df)  # X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)
    x_mean = scaler.mean_
    x_std = scaler.scale_
    # feature_df /= x_std
    return X


# agmuntations

## scaling

def scaling(true_featurs, corrent_featurs, corrent_labels, true_label, scale=3):
    # duplicate labels
    corrent_labels = pd.DataFrame(corrent_labels, columns=['class'])
    true_label = pd.DataFrame(true_label, columns=['class'])

    true_featurs = pd.DataFrame(true_featurs, columns=items)
    corrent_featurs = pd.DataFrame(corrent_featurs, columns=items)
    corrent_labels = corrent_labels.append(true_label, ignore_index=True)
    new_feature_df = true_featurs * scale
    corrent_featurs = corrent_featurs.append(new_feature_df, ignore_index=True)

    return corrent_featurs, corrent_labels


## flipping
def flip(true_featurs, corrent_featurs, corrent_labels, true_label):
    # duplicate labels
    corrent_labels = pd.DataFrame(corrent_labels, columns=['class'])
    true_label = pd.DataFrame(true_label, columns=['class'])
    corrent_labels = corrent_labels.append(true_label, ignore_index=True)

    true_featurs = pd.DataFrame(true_featurs, columns=items)
    corrent_featurs = pd.DataFrame(corrent_featurs, columns=items)
    new_feature_df = pd.DataFrame(columns=items)
    # true_featurs = true_featurs.rolling(window=9901).flip()

    for i in range(4):
        temp_df = true_featurs.loc[int(i * true_featurs.shape[0] / 4):int((i + 1) * true_featurs.shape[0] / 4 - 1), :].copy()

        temp_df = temp_df.iloc[::-1, :]

        new_feature_df = new_feature_df.append(temp_df, ignore_index=True)

    # new_feature_df = true_featurs.iloc[::-1, :].reset_index(drop=True)

    corrent_featurs = corrent_featurs.append(new_feature_df, ignore_index=True)

    return corrent_featurs, corrent_labels

def rotation(true_featurs, corrent_featurs, corrent_labels, true_label):
    # duplicate labels
    corrent_labels = pd.DataFrame(corrent_labels, columns=['class'])
    true_label = pd.DataFrame(true_label, columns=['class'])
    corrent_labels = corrent_labels.append(true_label, ignore_index=True)

    true_featurs = pd.DataFrame(true_featurs, columns=items)
    corrent_featurs = pd.DataFrame(corrent_featurs, columns=items)
    new_feature_df = pd.DataFrame(columns=items)
    # true_featurs = true_featurs.rolling(window=9901).flip()

    for i in range(4):
        temp_df = true_featurs.loc[int(i * true_featurs.shape[0] / 4):int((i + 1) * true_featurs.shape[0] / 4 - 1), :].copy()
        temp_df = temp_df.reset_index(drop=True)
        for index, item in temp_df.items():
           temp_df.loc[:,index] = agmontation.rotation(item).copy()


        new_feature_df = new_feature_df.append(temp_df, ignore_index=True)

    corrent_featurs = corrent_featurs.append(new_feature_df, ignore_index=True)

    return corrent_featurs, corrent_labels

##Permutation

def permutation(true_featurs, corrent_featurs, corrent_labels, true_label):
    # duplicate labels
    corrent_labels = pd.DataFrame(corrent_labels, columns=['class'])
    true_label = pd.DataFrame(true_label, columns=['class'])
    corrent_labels = corrent_labels.append(true_label, ignore_index=True)

    true_featurs = pd.DataFrame(true_featurs, columns=items)
    corrent_featurs = pd.DataFrame(corrent_featurs, columns=items)

    new_feature_df = pd.DataFrame(columns=items)
    for i in range(4):
        temp_df = true_featurs.loc[int(i * true_featurs.shape[0]/ 4):int((i + 1) * true_featurs.shape[0] / 4 - 1), :]
        nrows = temp_df.shape[0]
        b = np.random.permutation(nrows)
        temp_df = temp_df.take(b)
        new_feature_df = new_feature_df.append(temp_df, ignore_index=True)

    corrent_featurs = corrent_featurs.append(new_feature_df, ignore_index=True)

    return corrent_featurs, corrent_labels

def magnitude_wrap(true_featurs, corrent_featurs, corrent_labels, true_label):
    # duplicate labels
    corrent_labels = pd.DataFrame(corrent_labels, columns=['class'])
    true_label = pd.DataFrame(true_label, columns=['class'])
    corrent_labels = corrent_labels.append(true_label, ignore_index=True)

    true_featurs = pd.DataFrame(true_featurs, columns=items)
    corrent_featurs = pd.DataFrame(corrent_featurs, columns=items)

    new_feature_df = pd.DataFrame(columns=items)
    for i in range(4):
        temp_df = true_featurs.loc[int(i * true_featurs.shape[0] / 4):int((i + 1) * true_featurs.shape[0] / 4 - 1), :].copy()
        temp_df = temp_df.reset_index(drop=True)
        for index, item in temp_df.items():
           temp_df.loc[:,index] = agmontation.magnitude_warp(item).copy()

    corrent_featurs = corrent_featurs.append(new_feature_df, ignore_index=True)

    return corrent_featurs, corrent_labels


