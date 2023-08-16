import pandas as pd 
import matplotlib.pyplot as plt
import argparse
import numpy as np

import yaml

from data_proses import mask
with open(r'/home/robotics20/Documents/rotem/new_code/config.yaml', 'r') as f:
    args = yaml.safe_load(f)

config = argparse.Namespace(**args)

def plot_data(config,data):


    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,5))
    position = data.drop(['sesion_time_stamp'],axis=1)[config.positoin_label_inedx]

    ax1.plot(position)
    ax2.plot(data.drop(['sesion_time_stamp'],axis=1)[config.fmg_index])
    ax1.legend()

    plt.show() 


def is_not_numeric(x):
    try:
        float(x)
        return False
    except ValueError:
        return True
    

def print_not_numeric_vals(df):

    mask = df[config.fmg_index+config.first_positoin_label_inedx].applymap(is_not_numeric)
    non_numeric_values = df[mask].stack().dropna()
    print(non_numeric_values)

    return non_numeric_values    

def clean_data(df,not_numeric_vals):

    x = not_numeric_vals.index.to_numpy()
    index_list = [x[i][0] for i in range(len(x)) ]
    df = df.drop(index_list).reset_index(drop=True)
    df[config.fmg_index+config.first_positoin_label_inedx]=df[config.fmg_index+config.first_positoin_label_inedx].astype(np.float64)

    return df


if __name__== '__main__':


    df = pd.read_csv(r'/home/robotics20/Documents/rotem/new_code/data/data/16_Aug_2023_13_09.csv')
    df = df[config.fmg_index+config.first_positoin_label_inedx+config.sesion_time_stamp]   

    not_numeric_vals = print_not_numeric_vals(df)

    if not_numeric_vals.shape[0] == 0:
        plot_data(config=config,data=df)

    else :
        print("clean data")
        clean_df = clean_data(df,not_numeric_vals)
        clean_df = mask(clean_df,config=config)
        plot_data(config=config,data=clean_df)


    x = input("to save press 1\n ")

    if x == '1' :
        clean_df.to_csv(r'/home/robotics20/Documents/rotem/new_code/data/data/16_Aug_2023_13_09_clean.csv',index=False)