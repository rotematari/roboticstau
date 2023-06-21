import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt



import torch
from torch.optim import Adam

def train(config, train_loader, val_loader,net):
    # Create an instance of the FullyConnected class using the configuration object
    net = net(config)

    # Define the loss function
    criterion = net.rmsleloss()

    # Define the optimizer with weight decay
    optimizer = Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Create lists to store the training and validation loss 
    train_losses = []
    val_losses = []
    

    # Train the network
    for epoch in range(config.num_epochs):
        # Initialize the epoch loss and accuracy
        train_loss = 0

        # Train on the training set
        for inputs, targets in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the epoch loss and accuracy
            #TODO: add train_accuracy
            train_loss += loss.item()
           

        # Compute the average epoch loss and accuracy
        train_loss /= len(train_loader)
        

        # Save the epoch loss and accuracy values
        train_losses.append(train_loss)
        

        # Initialize the validation loss and accuracy
        val_loss = 0
        val_accuracy = 0

        # Evaluate on the validation set
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
            val_loss /= len(val_loader)
            

        # Save the validation loss and accuracy values
        val_losses.append(val_loss)
        
        # Print the epoch loss and accuracy values
        print(f'Epoch: {epoch} Train Loss: {train_loss}  Val Loss: {val_loss} ')

    return train_losses, val_losses


def test(net, config, test_loader):
    # Define the loss function
    criterion = net.rmsleloss()

    # Initialize the test loss and accuracy
    test_loss = 0
    test_accuracy = 0

    # Evaluate on the test set
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)

    return test_loss



def plot_losses(train_losses, val_losses=0,train=True):
    
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
    else:
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Plot the training and validation losses
        ax.plot(train_losses, label='Test Loss')
        
        # Add labels and a legend
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Show the plot
        plt.show() 




