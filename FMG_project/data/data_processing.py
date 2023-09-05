
import torch
from torch import Tensor
import pandas as pd
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from utils.utils import data_loder,mask,get_label_axis,calc_velocity,subtract_bias,sepatare_data,create_sliding_sequences,min_max_normalize

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = None


    def load_data(self)-> Tensor:
        config = self.config 

        self.data = data_loder(self.config)

        

    def preprocess_data(self):
            # drop over shot valuse 
            self.data = mask(self.data, self.config)
            # find zero axis 
            self.data[self.config.positoin_label_inedx] = get_label_axis(self.data[self.config.first_positoin_label_inedx], self.config)
            # adds vilocities to the labels
            self.data[self.config.velocity_label_inedx] = calc_velocity(self.config, self.data[self.config.first_positoin_label_inedx])
            # subtracts the bais on the FMG sensors 
            self.data[self.config.fmg_index] = subtract_bias(self.data[self.config.fmg_index+self.config.sesion_time_stamp])
            self.data = self.data.drop_duplicates().dropna().reset_index(drop=True)

            #normalize 
            self.data[self.config.positoin_label_inedx+self.config.velocity_label_inedx],label_max_val,label_min_val = min_max_normalize(self.data[self.config.positoin_label_inedx+self.config.velocity_label_inedx])
            self.data[self.config.fmg_index],fmg_max_val,fmg_min_val = min_max_normalize(self.data[self.config.fmg_index])

            #avereg rolling window
            self.data[self.config.fmg_index] = self.data[self.config.fmg_index].rolling(window=self.config.window_size, axis=0).mean()
            self.data[self.config.positoin_label_inedx+self.config.velocity_label_inedx] = self.data[self.config.positoin_label_inedx+self.config.velocity_label_inedx].rolling(window=self.config.window_size, axis=0).mean()



    def get_data_loaders(self):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return None, None

        fmg_df = self.data[self.config.fmg_index]
        label_df = self.data[self.config.positoin_label_inedx+self.config.velocity_label_inedx]
        
        # Creating sequences
        fmg_sequences = create_sliding_sequences(torch.tensor(fmg_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)
        label_sequences = create_sliding_sequences(torch.tensor(label_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)

        # Split the data into training and test sets
        train_fmg, test_fmg, train_label, test_label = train_test_split(fmg_sequences, label_sequences, test_size=self.config.test_size, random_state=None)

        # Split the training data into training and validation sets
        train_fmg, val_fmg, train_label, val_label = train_test_split(train_fmg, train_label, test_size=self.config.val_size / (1 - self.config.test_size), random_state=None)

        train_dataset = TensorDataset(train_fmg[:train_label.shape[0],:], train_label)
        val_dataset = TensorDataset(val_fmg[:val_label.shape[0],:], val_label)
        test_dataset = TensorDataset(test_fmg[:test_label.shape[0],:], test_label)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader, test_loader
    
