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
        

        size = last_in*multiplier

        if size >= 300 :
            size = 300
        hidden_size.append(int(size))
        last_in = int(last_in*multiplier)

    for layer in range(down):

        hidden_size.append(int(last_in/multiplier))
        last_in = int(last_in/multiplier)

    return hidden_size


def train(config, train_loader, val_loader,net,device='cpu',wandb_on=0):
    # Create an instance of the FullyConnected class using the configuration object
    # net = net(config)

    # Define the loss function
    criterion = net.mseloss

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
            for i,(inputs, targets) in enumerate(val_loader):
                
                inputs = inputs.to(device=device)
                targets = targets.to(device=device)

                outputs = net(inputs)
                v_loss = criterion(outputs, targets)
                val_loss += v_loss.item()
                # if v_loss.item()>1:
                #     print(v_loss.item())
                
                
                
            val_loss /= i
            

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
    criterion = net.mseloss

    # Initialize the test loss and accuracy
    test_loss = 0
    net.eval()

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

def threePointDist(outputs,targets):
# show distance between ground truth and prediction by 3d points [0,M2,M3,M4] 
# 'M1x', 'M1y','M1z', 'M2x', 'M2y', 'M2z', 'M3x', 'M3y', 'M3z', 'M4x', 'M4y', 'M4z',

   
    with torch.no_grad():
        dist = []
        outputs = outputs.cpu()
        targets = targets.cpu()
        M_predicted =np.array(outputs[:,0:0+3])
        M = np.array(targets[:,0:0+3])

        for i in range(2):

            M = np.concatenate([M,np.array(targets[:,i+3:i+6])],axis =1)
            M_predicted = np.concatenate([M_predicted,np.array(outputs[:,i+3:i+6])],axis =1)

        dist = np.sqrt((M[:,0]-M_predicted[:,0])**2+(M[:,1]-M_predicted[:,1])**2+(M[:,2]-M_predicted[:,2])**2)
        
        for i in range(2):

            dist = np.concatenate([dist,np.sqrt((M[:,i+3]-M_predicted[:,i+3])**2+(M[:,i+4]-M_predicted[:,i+4])**2+(M[:,i+5]-M_predicted[:,i+5])**2)],axis =0)
        dist = dist.reshape(-1,3)
    return dist.mean(axis=0)

def model_eval_metric(config,net,test_loader,device='cpu'):
    # show distance between ground truth and prediction by 3d points [0,M2,M3,M4] 

    net = net.to(device=device)
    net.eval()
    # Evaluate on the test set
    with torch.no_grad():

        inputs = test_loader.dataset.tensors[0]
        targets = test_loader.dataset.tensors[1]

        inputs = inputs.to(device=device)
        targets = targets.to(device=device)

        outputs = net(inputs)


        dist = threePointDist(outputs, targets)



    return dist



# normalization
def normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)


    return (data - mean) / std

def plot_data(config,data):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,5))

    ax1.plot(data.drop(['sesion_time_stamp'],axis=1)[config.positoin_label_inedx])
    ax2.plot(data.drop(['sesion_time_stamp'],axis=1)[config.fmg_index])
    ax1.legend()

    plt.show() 

if __name__== '__main__':

    hidden = hidden_size_maker(config=config)
    print(hidden) 

