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
    return full_df

def sepatare_data(full_df):


    imu_index = ['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz']
    fmg_index = ['S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20,S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35, S36, S37, S38, S39, S40, S41, S42, S43, S44, S45, S46, S47, S48']
    label_inedx = ['M1x,M1y,M1z,M2x,M2y,M2z,M3x,M3y,M3z,M4x,M4y,M4z']
    sesion_time_stamp = ['sesion_time_stamp']


    sesion_time_stamp_df = full_df['sesion_time_stamp']
    fmg_df = pd.concat([full_df[fmg_index],full_df['sesion_time_stamp']],axis=1,ignore_index=True)
    imu_df = pd.concat([full_df[imu_index],full_df['sesion_time_stamp']],axis=1,ignore_index=True)
    label_df = pd.concat([full_df[label_inedx],full_df['sesion_time_stamp']],axis=1,ignore_index=True)
    


    assert isinstance(fmg_df, pd.DataFrame),"fmg_df is not pd.dataframe"
    assert isinstance(imu_df, pd.DataFrame),"imu_df is not pd.dataframe"
    

    return fmg_df, imu_df, label_df  


## will find bias for each time stamped sesion 
def find_bias(fmg_df):

    bias_df = pd.DataFrame()

    for time_stamp in fmg_df['sesion_time_stamp'].unique():

        temp_df = fmg_df[fmg_df['sesion_time_stamp'] == time_stamp].mean().transpose()
        temp_df['sesion_time_stamp'] = time_stamp
        fmg_bias_df = pd.concat([fmg_bias_df,temp_df],axis= 1,ignore_index=True)
        fmg_bias_df = fmg_bias_df.transpose()    
    
    return fmg_bias_df

def find_std(fmg_df):
    
    std_df = pd.DataFrame()

    for time_stamp in fmg_df['sesion_time_stamp'].unique():

        temp_df = fmg_df[fmg_df['sesion_time_stamp'] == time_stamp].mean().transpose()
        temp_df['sesion_time_stamp'] = time_stamp
        fmg_bias_df = pd.concat([fmg_bias_df,temp_df],axis= 1,ignore_index=True)
        fmg_bias_df = fmg_bias_df.transpose()    
    
    return fmg_bias_df


   

def subtract_bias(fmg_df):

    bias_df = find_bias(fmg_df)
    for time_stamp in fmg_df['sesion_time_stamp'].unique():
            
        temp_fmg_df = fmg_df[fmg_df['sesion_time_stamp'] ==time_stamp].drop('sesion_time_stamp',axis=1) - np.array(bias_df[bias_df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1))
        new_fmg_df = pd.concat([new_fmg_df,temp_fmg_df],axis= 0)

    return new_fmg_df


def std_division(fmg_df):

    std_df = find_std(fmg_df)
    for time_stamp in fmg_df['sesion_time_stamp'].unique():
            
        temp_fmg_df = fmg_df[fmg_df['sesion_time_stamp'] ==time_stamp].drop('sesion_time_stamp',axis=1) - np.array(std_df[std_df['sesion_time_stamp'] == time_stamp].drop('sesion_time_stamp',axis=1))
        new_fmg_df = pd.concat([new_fmg_df,temp_fmg_df],axis= 0)

    return new_fmg_df
 


