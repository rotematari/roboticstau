
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join



def data_loder(dir_path):
    full_df = pd.DataFrame()
    for file in listdir(dir_path):

        df = pd.read_csv(join(dir_path,file))
        full_df = pd.concat([full_df,df],axis=0,ignore_index=True)
        full_df = full_df.replace(-np.inf, np.nan)
        full_df = full_df.replace(np.inf, np.nan)
        full_df = full_df.iloc[1:,:-1].dropna(axis=0)
        


    return full_df

def sepatare_data(full_df):


    imu_index = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz']
    fmg_index = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20','S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46', 'S47', 'S48']
    label_inedx = ['M1x','M1y','M1z','M2x','M2y','M2z','M3x','M3y','M3z','M4x','M4y','M4z']
    sesion_time_stamp = ['sesion_time_stamp']


    sesion_time_stamp_df = full_df['sesion_time_stamp']
    fmg_df = pd.concat([full_df[fmg_index],full_df['sesion_time_stamp']],axis=1,ignore_index=False)
    # imu_df = pd.concat([full_df[imu_index],full_df['sesion_time_stamp']],axis=1,ignore_index=True)
    label_df = pd.concat([full_df[label_inedx],full_df['sesion_time_stamp']],axis=1,ignore_index=False)
    


    assert isinstance(fmg_df, pd.DataFrame),"fmg_df is not pd.dataframe"
    # assert isinstance(imu_df, pd.DataFrame),"imu_df is not pd.dataframe"
    imu_df=0

    return fmg_df, imu_df, label_df 


## will find bias for each time stamped sesion 
def find_bias(df):

    bias_df = pd.DataFrame()

    for time_stamp in df['sesion_time_stamp'].unique():

        temp_df = pd.DataFrame(df[df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1),dtype=float).mean().to_frame().T
        temp_df['sesion_time_stamp'] = time_stamp
        bias_df = pd.concat([bias_df,temp_df],axis= 0,ignore_index=False)
            
    
    return bias_df

def find_std(df):
    
    std_df = pd.DataFrame()

    for time_stamp in df['sesion_time_stamp'].unique():

        temp_df = pd.DataFrame(df[df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1),dtype=float).std().to_frame().T
        temp_df['sesion_time_stamp'] = time_stamp
        std_df = pd.concat([std_df,temp_df],axis=0,ignore_index=False)
           
    
    return std_df


   

def subtract_bias(df):

    bias_df = find_bias(df)
    new_df = pd.DataFrame()
    for time_stamp in df['sesion_time_stamp'].unique():
            
        temp_df = pd.DataFrame(df[df['sesion_time_stamp'] ==time_stamp].drop('sesion_time_stamp',axis=1).astype(float) - np.array(bias_df[bias_df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1).astype(float)))
        temp_df['sesion_time_stamp'] = time_stamp
        new_df = pd.concat([new_df,temp_df],axis= 0,ignore_index=False)

    return new_df


def std_division(df):
    new_fmg_df = pd.DataFrame()
    std_df = find_std(df)
    for time_stamp in df['sesion_time_stamp'].unique():
            
        temp_df = df[df['sesion_time_stamp'] ==time_stamp].drop('sesion_time_stamp',axis=1).astype(float) / np.array(std_df[std_df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1).astype(float))
        temp_df['sesion_time_stamp'] = time_stamp
        new_fmg_df = pd.concat([new_fmg_df,temp_df],axis= 0)

    return new_fmg_df
 

# def plot_data(df):


    





# ## torch data set 
# class data(Dataset):
#     def __init__(self,args_config,train=True):

#         ## get dir names and location 


#         ## seperates the IMU data from the FGM data 
#         # make sure each one has labalings 

        


#         ## finds the mean/bias of each data collection sesion - calclulated from the relaxed data 

#         ## subtracts the bias from the data 



#     def __getitem__(self, index):
#         return self.featurs[index],self.labels[index]

#     def __len__(self):
#         return self.n_samples   
