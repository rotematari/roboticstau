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
from torch.nn import L1Loss

<<<<<<< HEAD
<<<<<<< Updated upstream
def train(config, train_loader, val_loader,net):
=======
=======
>>>>>>> main
import wandb
import argparse


import time
<<<<<<< Updated upstream
<<<<<<< HEAD
import sys
sys.path.insert(0, r'new_code')
import yaml


with open(r'C:\Users\rotem\OneDrive\Desktop\main\roboticstau\new_code\config.yaml', 'r') as f:
    
=======
import yaml
with open('/home/robotics20/Documents/rotem/new_code/config.yaml', 'r') as f:
>>>>>>> main
=======
import sys
sys.path.insert(0, r'new_code')
import yaml


with open(r'C:\Users\rotem\OneDrive\Desktop\main\roboticstau\new_code\config.yaml', 'r') as f:
    
>>>>>>> Stashed changes
    args = yaml.safe_load(f)

config = argparse.Namespace(**args)


def hidden_size_maker(config,seq=True):
    hidden_size = []
    if seq:
        last_in = config.seq_size
    else :
        last_in = config.input_size

    
    up = int(config.n_layer/2)
    down = config.n_layer -up 
    multiplier = config.multiplier
    for layer in range(up):
        

        size = last_in*multiplier

        if size >= 2560 :
            size = 2560



        hidden_size.append(int(size))
        last_in = int(last_in*multiplier)

    for layer in range(down):
        size = last_in/multiplier
        
        if size<=config.label_seq_size*2:
            size = config.label_seq_size*2

        hidden_size.append(int(size))
        last_in = int(size)

    return hidden_size


def train(config, train_loader, val_loader,net,device='cpu',wandb_on=0):
<<<<<<< HEAD
>>>>>>> Stashed changes
=======
>>>>>>> main
    # Create an instance of the FullyConnected class using the configuration object
    # net = net(config)

    # Define the loss function
    criterion = L1Loss() 


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

        M_predicted =np.array(outputs[:,:3])
        M = np.array(targets[:,:3])

        for i in range(2):

            M = np.concatenate([M,np.array(targets[:,i+3:i+6])],axis =1)
            M_predicted = np.concatenate([M_predicted,np.array(outputs[:,i+3:i+6])],axis =1)

        dist = np.sqrt((M[:,0]-M_predicted[:,0])**2+(M[:,1]-M_predicted[:,1])**2+(M[:,2]-M_predicted[:,2])**2)
        
        for i in range(2):

            dist = np.concatenate([dist,np.sqrt((M[:,i+3]-M_predicted[:,i+3])**2+(M[:,i+4]-M_predicted[:,i+4])**2+(M[:,i+5]-M_predicted[:,i+5])**2)],axis =0)
        dist = dist.reshape(-1,3)
    return dist.mean(axis=0)

def model_eval_metric(config,net,test_loader,label_max_val,label_min_val ,device='cpu'):
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

        # Create seq_length series of shape (seq_length*num_labels,)
        label_min_val = [label_min_val for _ in range(config.seq_length)]
        label_max_val = [label_max_val for _ in range(config.seq_length)]

        # Concatenate the series to create a new series of shape (180,)
        new_label_min_val = np.concatenate(label_min_val)
        new_label_max_val = np.concatenate(label_max_val)



        outputs = min_max_unnormalize(outputs.detach().cpu().numpy(),new_label_min_val,new_label_max_val)
        outputs = torch.tensor(outputs)
        dist = threePointDist(outputs.reshape(-1,18), targets.reshape(-1,18))



    return dist

def min_max_unnormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    norm = (data - min_val) / (max_val - min_val)
    return norm,max_val,min_val


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

def rollig_window(config,data):

    data_avg =data.copy()
    data_avg[config.fmg_index] = data_avg[config.fmg_index].rolling(window=config.window_size, axis=0).mean()

    return data_avg 
# if __name__== '__main__':

    

