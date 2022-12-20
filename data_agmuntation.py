from os import listdir
from os.path import join, isfile

import pandas as pd
from sklearn.preprocessing import StandardScaler


# finds the mean of relaxed state

def find_mean(states_dir, data_dir_path, items):
    for dir_name in states_dir:
        filepath = data_dir_path + '/' + dir_name
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
