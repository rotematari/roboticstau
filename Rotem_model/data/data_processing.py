import torch
from torch import Tensor
import pandas as pd
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os 

# # Change the current working directory to the directory of the main script
# os.chdir(join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from sklearn.model_selection import train_test_split
from utils.utils import data_loader, mask, get_label_axis, calc_velocity, subtract_bias, create_sliding_sequences, min_max_normalize

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.label_max_val = None
        self.label_min_val = None
        self.fmg_max_val = None
        self.fmg_min_val = None
        self.label_size = config.num_labels

        if config.with_velocity:
            self.label_index = config.position_label_index + config.velocity_label_index
            self.label_size = config.num_labels
        else:
            self.label_index = config.position_label_index
            self.label_size = len(config.position_label_index)

    def load_data(self) -> Tensor:
        config = self.config
        self.data = data_loader(config)

    def preprocess_data(self):
        # drop overshot values
        self.data = mask(self.data, self.config)

        # find zero axis
        self.data[self.config.position_label_index] = get_label_axis(self.data[self.config.first_position_label_index], self.config)

        if self.config.with_velocity:
            # adds velocities to the labels
            self.data[self.label_index] = calc_velocity(self.config, self.data[self.config.first_position_label_index])

        # subtracts the bias on the FMG sensors data
        self.data[self.config.fmg_index] = subtract_bias(self.data[self.config.fmg_index + self.config.session_time_stamp])
        self.data = self.data.drop_duplicates().dropna().reset_index(drop=True)

        # normalize
        if self.config.norm_labels:
            self.data[self.label_index], self.label_max_val, self.label_min_val = min_max_normalize(self.data[self.label_index])

        self.data[self.config.fmg_index], self.fmg_max_val, self.fmg_min_val = min_max_normalize(self.data[self.config.fmg_index])

        # average rolling window
        self.data[self.config.fmg_index] = self.data[self.config.fmg_index].rolling(window=self.config.window_size, axis=0).mean()
        self.data[self.label_index] = self.data[self.label_index].rolling(window=self.config.window_size, axis=0).mean()

        self.data = self.data.drop_duplicates().dropna().reset_index(drop=True)

    def get_data_loaders(self):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return None, None

        fmg_df = self.data[self.config.fmg_index]
        label_df = self.data[self.label_index]

        if self.config.sequence:
            # Creating sequences
            features = create_sliding_sequences(torch.tensor(fmg_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)
            labels = create_sliding_sequences(torch.tensor(label_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)
        else:
            features = torch.tensor(fmg_df.to_numpy(), dtype=torch.float32)
            labels = torch.tensor(label_df.to_numpy(), dtype=torch.float32)

        # Split the data into training and test sets
        train_fmg, test_fmg, train_label, test_label = train_test_split(features, labels, test_size=self.config.test_size, random_state=None, shuffle=False)

        # Split the training data into training and validation sets
        train_fmg, val_fmg, train_label, val_label = train_test_split(train_fmg, train_label, test_size=self.config.val_size / (1 - self.config.test_size), random_state=self.config.random_state, shuffle=self.config.shuffle_train)

        train_dataset = TensorDataset(train_fmg[:train_label.shape[0], :], train_label)
        val_dataset = TensorDataset(val_fmg[:val_label.shape[0], :], val_label)
        test_dataset = TensorDataset(test_fmg[:test_label.shape[0], :], test_label)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader, test_loader

    def plot(self, from_indx=0, to_indx=1500):
        fmg_df = self.data[self.config.fmg_index]
        label_position = self.data[self.config.position_label_index]

        if self.config.with_velocity:
            label_velocity = self.data[self.config.velocity_label_index]

        # Create a figure and a grid of subplots with 1 row and 2 columns
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)  # Adjust figsize as needed

        # Plot data on the second subplot
        axes[0].plot(fmg_df[from_indx:to_indx])
        axes[0].set_title('Plot of FMG')

        axes[1].plot(label_position[from_indx:to_indx])
        axes[1].set_title('Plot of label_position')

        if self.config.with_velocity:
            axes[2].plot(label_velocity[from_indx:to_indx])
            axes[2].set_title('Plot of label_velocity')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()
        return
