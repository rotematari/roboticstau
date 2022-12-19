from os import listdir
from os.path import join, isfile

import pandas as pd
from sklearn.preprocessing import StandardScaler


# finds the mean of relaxed state

def find_mean(filesanddir, dirpath, items):
    for dir_name in filesanddir:
        filepath = dirpath + '/' + dir_name
        onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        # clac mean val for relaxed state in each data acquisition session
        if dir_name == 'relaxed':
            df_mean = pd.DataFrame()
            c = 0
            for file_name in onlyfiles:
                df = pd.read_csv(join(filepath, file_name))
                df = df.iloc[100:, :].reset_index(drop=True)
                df.drop(['time'], axis=1, inplace=True, errors="ignor")
                df = df.filter(items=items)
                num = file_name[file_name.find('_') + 1]
                df_mean[num] = df.mean()  # new data frame of mean vals

    return df_mean


# 1 filter the data by 10 point avg
def filter(feature_df):
    feature_df_roll = feature_df.rolling(window=100).mean()
    feature_df_roll = feature_df_roll.dropna().reset_index(drop=True)

    return feature_df_roll


# min max normalization
def min_max_norm(feature_df):
    df_norm = (feature_df - feature_df.min()) / (feature_df.max() - feature_df.min())

    return df_norm


# standardization

def stdnorm(feaure_df):
    scaler = StandardScaler()
    scaler.fit(feaure_df)
    X = scaler.transform(feaure_df)  # X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)
    x_mean = scaler.mean_
    x_std = scaler.scale_
    return X
