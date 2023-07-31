
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

def data_loder(config):
    """
    Given a directory path, loads all the csv files in the directory and returns a concatenated pandas dataframe.
    """
    full_df = pd.DataFrame()
    for file in listdir(config.data_path):

        df = pd.read_csv(join(config.data_path,file))
        full_df = pd.concat([full_df,df],axis=0,ignore_index=True)
        full_df = full_df.replace(-np.inf, np.nan)
        full_df = full_df.replace(np.inf, np.nan)
        # full_df = full_df.iloc[1:,:-1].dropna(axis=0)

    return full_df

def sepatare_data(full_df,config,first=True):
    """
    Given a pandas dataframe containing FMG, IMU and label data, separates the data into three pandas dataframes: 
    one for FMG data, one for IMU data and one for label data.
    """
    # # imu_index = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz']
    # fmg_index = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16']
    label_inedx = config.first_positoin_label_inedx
    # sesion_time_stamp = ['sesion_time_stamp']
    if first:
        # sesion_time_stamp_df = full_df[config.sesion_time_stamp]
        fmg_df = pd.concat([full_df[config.fmg_index],full_df[config.sesion_time_stamp]],axis=1,ignore_index=False)
        # imu_df = pd.concat([full_df[imu_index],full_df['sesion_time_stamp']],axis=1,ignore_index=True)
        label_df = pd.concat([full_df[label_inedx],full_df[config.sesion_time_stamp]],axis=1,ignore_index=False)
    else:


        fmg_df = pd.concat([full_df[config.fmg_index],full_df[config.sesion_time_stamp]],axis=1,ignore_index=False)
        # imu_df = pd.concat([full_df[imu_index],full_df['sesion_time_stamp']],axis=1,ignore_index=True)
        label_df = pd.concat([full_df[config.positoin_label_inedx+config.velocity_label_inedx],full_df[config.sesion_time_stamp]],axis=1,ignore_index=False)

    assert isinstance(fmg_df, pd.DataFrame),f"{fmg_df} is not a DataFrame"
    assert isinstance(label_df, pd.DataFrame),f"{label_df} is not a DataFrame"
    # assert isinstance(imu_df, pd.DataFrame),"imu_df is not pd.dataframe"
    imu_df=0

    return fmg_df, imu_df, label_df 

## will find bias for each time stamped sesion 
def find_bias(df):
    """
    Given a pandas dataframe containing FMG or IMU data, finds the bias for each time stamped session and returns a pandas dataframe.
    """
    bias_df = pd.DataFrame()

    for time_stamp in df['sesion_time_stamp'].unique():

        temp_df = pd.DataFrame(df[df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1),dtype=float).mean().to_frame().T
        temp_df['sesion_time_stamp'] = time_stamp
        bias_df = pd.concat([bias_df,temp_df],axis= 0,ignore_index=False)

    return bias_df



def find_std(df):
    """
    Given a pandas dataframe containing FMG or IMU data, finds the standard deviation for each time stamped session and returns a pandas dataframe.
    """
    std_df = pd.DataFrame()

    for time_stamp in df['sesion_time_stamp'].unique():

        temp_df = df[df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1).std().T.copy()
        temp_df['sesion_time_stamp'] = time_stamp
        std_df = pd.concat([std_df,temp_df],axis=1,ignore_index=False)

    return std_df.T


def subtract_bias(df):
    # Compute the bias for each unique value of the sesion_time_stamp column
    bias_df = find_bias(df)
    
    # Initialize an empty DataFrame to store the result
    new_df = pd.DataFrame()
    
    # Iterate over each unique value of the sesion_time_stamp column
    for time_stamp in df['sesion_time_stamp'].unique():
        # Select the rows of df and bias_df corresponding to the current time stamp
        df_rows = df[df['sesion_time_stamp'] == time_stamp].copy()
        bias_rows = bias_df[bias_df['sesion_time_stamp'] == time_stamp].copy()
        
        df_rows= df_rows.drop('sesion_time_stamp', axis=1).astype(float).copy()
        bias_rows = bias_rows.drop('sesion_time_stamp', axis=1).astype(float).copy()
        
        # Subtract the bias from the data in df
        temp_df = df_rows-bias_rows.to_numpy() 
        

        # Add back the sesion_time_stamp column
        temp_df['sesion_time_stamp'] = time_stamp
        
        # Append the result to new_df
        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=False)
    
    return new_df



def std_division(df):
    # Compute the standard deviation for each unique value of the sesion_time_stamp column
    std_df = find_std(df)
    
    # Initialize an empty DataFrame to store the result
    new_fmg_df = pd.DataFrame()
    
    # Iterate over each unique value of the sesion_time_stamp column
    for time_stamp in df['sesion_time_stamp'].unique():
        # Select the rows of df and std_df corresponding to the current time stamp
        df_rows = df[df['sesion_time_stamp'] == time_stamp].copy()
        std_rows = std_df[std_df['sesion_time_stamp'] == time_stamp].copy()
        
        df_rows= df_rows.drop('sesion_time_stamp', axis=1).astype(float).copy()
        std_rows = std_rows.drop('sesion_time_stamp', axis=1).astype(float).copy()


        
        # Divide the data in df by the standard deviation
        temp_df = df_rows/ (std_rows.to_numpy()+1e-4)
        
        # Add back the sesion_time_stamp column
        temp_df['sesion_time_stamp'] = time_stamp
        
        # Append the result to new_fmg_df
        new_fmg_df = pd.concat([new_fmg_df, temp_df], axis=0)
    
    return new_fmg_df


def get_label_axis(labels,config):
    #label_inedx = ['M1x','M1y','M1z','M2x','M2y','M2z','M3x','M3y','M3z','M4x','M4y','M4z']
    
    x  = labels[['M1x','M2x','M3x','M4x']].sub(labels['M1x'], axis=0)
    y = labels[['M1y','M2y','M3y','M4y']].sub(labels['M1y'], axis=0)
    z = labels[['M1z','M2z','M3z','M4z']].sub(labels['M1z'], axis=0)
                    


    new_labels = pd.concat((x,y,z),axis=1)
    new_labels['sesion_time_stamp'] = labels['sesion_time_stamp']
    return new_labels[config.positoin_label_inedx]



def calc_velocity(config,label_df):
#['V2x','V2y','V2z','V3x','V3y','V3z','V4x','V4y','V4z']

    label_df[config.velocity_label_inedx]= [0,0,0,0,0,0,0,0,0]
    temp = label_df.loc[1:,config.positoin_label_inedx].reset_index(drop=True).copy()
    
    label_df.loc[:temp.shape[0]+2,config.velocity_label_inedx] = temp.values - label_df.loc[:temp.shape[0]+2,config.positoin_label_inedx].values
    return label_df

def mask(data,config):

    # create a mask that selects rows where the values in fmg_index columns are greater than 1024
    mask1 = (data[config.fmg_index] > 1024).any(axis=1)

    # create a mask that selects rows where the values in first_position_label_index columns are greater than 2
    mask2 = (data[config.first_positoin_label_inedx] > 2).any(axis=1)

    # combine the masks using the | (or) operator
    mask = mask1 | mask2

    # drop the rows from the DataFrame
    data = data.drop(data[mask].index)


    return data

def drop_nonnumber_val(data,config):



    return data 

