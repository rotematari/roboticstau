import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

import sys
import os

import torch
from torch.optim import Adam

import wandb
import argparse
import yaml

import time

with open('/home/robotics20/Documents/rotem/new_code/config.yaml', 'r') as f:
    args = yaml.safe_load(f)

config = argparse.Namespace(**args)


def hidden_size_maker(config):
    hidden_size = []
    last_in = config.input_size
    up = int(config.n_layer/2)
    down = config.n_layer -up 
    multiplier = config.multiplier
    for layer in range(up):
        
        hidden_size.append(int(last_in*multiplier))
        last_in = int(last_in*multiplier)

    for layer in range(down):

        hidden_size.append(int(last_in/multiplier))
        last_in = int(last_in/multiplier)

    return hidden_size


def train(config, train_loader, val_loader,net,device='cpu',wandb_on=0):
    # Create an instance of the FullyConnected class using the configuration object
    # net = net(config)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the optimizer with weight decay
    optimizer = Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Create lists to store the training and validation loss 
    train_losses = []
    val_losses = []
    best_val_loss = 0

    # Train the network
    for epoch in range(config.num_epochs):
        # Initialize the epoch loss and accuracy
        train_loss = 0

        # Train on the training set
        for inputs, targets in train_loader:

            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the epoch loss and accuracy
            train_loss += loss.item()
           

        # Compute the average epoch loss and accuracy
        train_loss /= len(train_loader)
        

        # Save the epoch loss and accuracy values
        train_losses.append(train_loss)
        

        # Initialize the validation loss and accuracy
        val_loss = 0

        # Evaluate on the validation set
        with torch.no_grad():
            for inputs, targets in val_loader:
                
                inputs = inputs.to(device=device)
                targets = targets.to(device=device)

                outputs = net(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
            val_loss /= len(val_loader)
            

        # Save the validation loss and accuracy values
        val_losses.append(val_loss)
        
        # Print the epoch loss and accuracy values
        print(f'Epoch: {epoch} Train Loss: {train_loss}  Val Loss: {val_loss} ')
        if wandb_on:
            # log metrics to wandb
            wandb.log({"Train Loss": train_loss, "val_loss": val_loss})

        if(best_val_loss < val_loss):
            time_stamp = time.strftime("%d_%b_%Y_%H:%M", time.gmtime())
            best_val_loss = val_loss
            filename = str(epoch+1)+time_stamp + '.pt'
            checkpoint_path = join(config.model_path,filename)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                }, checkpoint_path)

    return train_losses, val_losses


def test(net, config, test_loader,device='cpu',wandb_on=0):
    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Initialize the test loss and accuracy
    test_loss = 0
    

    # Evaluate on the test set
    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        

    return test_loss



def plot_losses(train_losses, val_losses=[],train=True):
    
    if train:
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Plot the training and validation losses
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        
        
        # Add labels and a legend
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Show the plot
        plt.show()

if __name__== '__main__':

    hidden = hidden_size_maker(config=config)
    print(hidden) 

