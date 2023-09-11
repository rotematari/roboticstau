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
from torch.nn import L1Loss,MSELoss,HuberLoss
from torch import Tensor

import wandb
import argparse


import time
import yaml
# with open('config.yaml', 'r') as f:
#     args = yaml.safe_load(f)

# config = argparse.Namespace(**args)

# Change the current working directory to the directory of the main script
os.chdir(join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

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


def train(config, train_loader, val_loader,model,device='cpu',wandb_run=None):
    # Create an instance of the FullyConnected class using the configuration object
    # net = net(config)

    # Define the loss function
    criterion = MSELoss() 


    # Define the optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Create lists to store the training and validation loss 
    train_losses = []
    val_losses = []
    best_val_loss = 0
    print("training starts")

    # Train the network
    for epoch in range(config.num_epochs):
        # Initialize the epoch loss and accuracy
        train_loss = 0
        model.train()
        # Train on the training set
        for inputs, targets in train_loader:

            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            if config.sequence:
                loss = criterion(outputs, targets[:,-1:,:].view(-1,config.num_labels))
            else:
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
            model.eval()
            for i,(inputs, targets) in enumerate(val_loader):
                
                inputs = inputs.to(device=device)
                targets = targets.to(device=device)

                outputs = model(inputs)

            if config.sequence:
                v_loss = criterion(outputs, targets[:,-1:,:].view(-1,config.num_labels))
            else:
                v_loss = criterion(outputs, targets)
                val_loss += v_loss.item()
                
            val_loss /= i
            
        # Save the validation loss 
        val_losses.append(val_loss)
        
        # Print the epoch loss and accuracy values
        print(f'Epoch: {epoch} Train Loss: {train_loss}  Val Loss: {val_loss} ')
        if wandb_run is not None:
            # log metrics to wandb
            wandb_run.log({"Train Loss": train_loss, "Val_loss": val_loss})

        if(best_val_loss < val_loss):
            time_stamp = time.strftime("%d_%m_%Y_%H:%M", time.gmtime())
            best_val_loss = val_loss
            filename = 'epoch'+str(epoch+1)+'_'+time_stamp + '.pt'
            checkpoint_path = join(config.model_path,filename)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                }, checkpoint_path)
            
    time_stamp = time.strftime("%d_%m_%Y_%H:%M", time.gmtime())
    filename = 'epoch'+str(epoch+1)+'_'+time_stamp + '.pt'
    checkpoint_path = join(config.model_path,filename)
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        }, checkpoint_path)
    
    print(f"model {filename} saved ")
    
    return train_losses, val_losses


def test(model, config, test_loader,device='cpu',wandb_on=0):
    # Define the loss function
    criterion = model.mseloss

    # Initialize the test loss and accuracy
    test_loss = 0
    model.eval()

    # Evaluate on the test set
    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            outputs = model(inputs)
            loss = criterion(outputs, targets[:,-1:].squeeze())
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


def plot_results(config,preds,targets,wandb_run=None):

    # Create a figure and a grid of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(10, 4),sharex=True,sharey=True)  # Adjust figsize as needed

    # Plot data on the first subplot
    axes[0,0].plot(preds[:,:9])
    axes[0,0].set_title('Plot of preds location')
    # axes[0,0].legend()
    axes[0,1].plot(preds[:,9:])
    axes[0,1].set_title('Plot of preds V ')
    # axes[0,1].legend()

    # Plot data on the second subplot
    axes[1,0].plot(targets[:,:9])
    axes[1,0].set_title('Plot of targets location')

    if config.with_velocity:
        axes[1,1].plot(targets[:,9:])
        axes[1,1].set_title('Plot of targets V')
        

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if wandb_run is not None:
        # Log the figure to wandb
        wandb.log({"preds and targets": plt})
    else:
        # Show the plots
        plt.show()

def model_eval_metric(config,model,test_loader,label_max_val,label_min_val ,device='cpu',wandb_run=None):
    # show distance between ground truth and prediction by 3d points [0,M2,M3,M4] 

    model = model.to(device=device)
    model.eval()
    # Evaluate on the test set
    with torch.no_grad():

        inputs = test_loader.dataset.tensors[0]
        targets = test_loader.dataset.tensors[1]
        

        inputs = inputs.to(device=device)
        targets = targets.to(device=device)
        if config.sequence:
            outputs = model(inputs[:,-1:,:])
            size = outputs.size(0)
            
            if config.norm_labels:
                outputs = min_max_unnormalize(outputs.detach().cpu().numpy(),np.tile(label_min_val,(size,1)),np.tile(label_max_val,(size,1)))
                targets = min_max_unnormalize(targets.detach().cpu().numpy(),np.tile(label_min_val,(targets.size(0),targets.size(1),1)),np.tile(label_max_val,(targets.size(0),targets.size(1),1)))
           
            if config.plot_pred:
                plot_results(config,outputs[100:2000].cpu().detach().numpy(),targets[100:2000,-1:,:].view(-1,config.num_labels).cpu().detach().numpy(),wandb_run=wandb_run)

            dist = np.sqrt(((outputs.cpu().detach().numpy() - targets[:,-1:,:].view(-1,config.num_labels).cpu().detach().numpy())**2).sum(axis=0)/size)
        else:
            outputs = model(inputs[:2000])
            size = outputs.size(0)
            if config.norm_labels:
                outputs = min_max_unnormalize(outputs.detach().cpu().numpy(),np.tile(label_min_val,(size,1)),np.tile(label_max_val,(size,1)))
                targets = min_max_unnormalize(targets.detach().cpu().numpy(),np.tile(label_min_val,(targets.size(0),1)),np.tile(label_max_val,(targets.size(0),1)))

            if config.plot_pred:
                plot_results(config,outputs[100:2000].cpu().detach().numpy(),targets[100:2000,:].view(-1,config.num_labels).cpu().detach().numpy(),wandb_run=wandb_run)

            dist = np.sqrt(((outputs.cpu().detach().numpy() - targets[:size].cpu().detach().numpy())**2).sum(axis=0)/size)

    return dist

def min_max_unnormalize(data, min_val, max_val,bottom=-1, top=1):

    return torch.tensor(((data-bottom)/(top-bottom)) * (max_val - min_val) + min_val)

def min_max_normalize(data,bottom=-1, top=1):

    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    norm = bottom+(data - min_val) / (max_val - min_val)*(top-bottom)

    return norm,max_val,min_val


# normalization std
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

# clould be deleted 
def rollig_window(config,data):

    data_avg =data.copy()
    data_avg[config.fmg_index] = data_avg[config.fmg_index].rolling(window=config.window_size, axis=0).mean()

    return data_avg 

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
        label_df = pd.concat([full_df[label_inedx],full_df[config.sesion_time_stamp]],axis=1,ignore_index=False)
    else:


        fmg_df = pd.concat([full_df[config.fmg_index],full_df[config.sesion_time_stamp]],axis=1,ignore_index=False)

        label_df = pd.concat([full_df[config.positoin_label_inedx+config.velocity_label_inedx],full_df[config.sesion_time_stamp]],axis=1,ignore_index=False)

    assert isinstance(fmg_df, pd.DataFrame),f"{fmg_df} is not a DataFrame"
    assert isinstance(label_df, pd.DataFrame),f"{label_df} is not a DataFrame"
    # assert isinstance(imu_df, pd.DataFrame),"imu_df is not pd.dataframe"
    imu_df=0

    return fmg_df,label_df 

## will find bias for each time stamped sesion 
def find_bias(df):
    """
    Given a pandas dataframe containing FMG data, finds the bias for each time stamped session and returns a pandas dataframe.
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
        

        # # Add back the sesion_time_stamp column
        # temp_df['sesion_time_stamp'] = time_stamp
        
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
    # Create a copy of the labels DataFrame slice
    # labels_copy = labels[config.first_positoin_label_inedx].copy()
    labels = labels.copy()
    # Now perform the operations on the copy
    labels.loc[:,['M1x','M2x','M3x','M4x']]  = labels[['M1x','M2x','M3x','M4x']].sub(labels['M1x'], axis=0)
    labels.loc[:,['M1y','M2y','M3y','M4y']] = labels[['M1y','M2y','M3y','M4y']].sub(labels['M1y'], axis=0)
    labels.loc[:,['M1z','M2z','M3z','M4z']] = labels[['M1z','M2z','M3z','M4z']].sub(labels['M1z'], axis=0)

    # If you want to replace these columns in the original 'labels' DataFrame:
    # labels.loc[:,config.first_positoin_label_inedx] = labels_copy

#    labels[['M1x','M2x','M3x','M4x']]  = labels[['M1x','M2x','M3x','M4x']].sub(labels['M1x'], axis=0)
#    labels[['M1y','M2y','M3y','M4y']] = labels[['M1y','M2y','M3y','M4y']].sub(labels['M1y'], axis=0)
#    labels[['M1z','M2z','M3z','M4z']] = labels[['M1z','M2z','M3z','M4z']].sub(labels['M1z'], axis=0)
   
    return labels[config.positoin_label_inedx]


def calc_velocity(config,label_df):
    #['V2x','V2y','V2z','V3x','V3y','V3z','V4x','V4y','V4z']
    label_df = label_df.copy()
    label_df[config.velocity_label_inedx]= [0,0,0,0,0,0,0,0,0]
    temp = label_df.loc[1:,config.positoin_label_inedx].reset_index(drop=True).copy()

    label_df = label_df.loc[:temp.shape[0]-10] 
    temp = temp.loc[:temp.shape[0]-10]
    label_df.loc[:temp.shape[0],config.velocity_label_inedx] = temp.values - label_df.loc[:temp.shape[0],config.positoin_label_inedx].values
    return label_df[config.velocity_label_inedx]

def calc_velocity(config, label_df):
    # Copy the dataframe to avoid SettingWithCopyWarning
    label_df = label_df.copy()
    
    # Time interval in seconds, based on 60 Hz frequency
    delta_t = 1 / config.sample_speed  
    
    # Columns corresponding to positions
    position_cols = config.positoin_label_inedx
    
    # Columns corresponding to velocities
    velocity_cols = config.velocity_label_inedx
    
    # Calculate velocity
    velocity_df = label_df[position_cols].diff() / delta_t
    velocity_df.columns = velocity_cols
    # Update the original dataframe with the calculated velocities
    label_df[velocity_cols] = velocity_df

    #     # Plot the first 500 samples
    # if len(label_df) >= 500:
    #     fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    #     # Plot position data
    #     axes[0].plot(label_df.iloc[:500][position_cols])
    #     axes[0].set_title('Position Data')
    #     axes[0].set_xlabel('Sample Index')
    #     axes[0].set_ylabel('Position')
    #     axes[0].legend(position_cols)

    #     # Plot velocity data
    #     axes[1].plot(label_df.iloc[:500][velocity_cols])
    #     axes[1].set_title('Velocity Data')
    #     axes[1].set_xlabel('Sample Index')
    #     axes[1].set_ylabel('Velocity')
    #     axes[1].legend(velocity_cols)

    #     plt.tight_layout()
    #     plt.show()

    return label_df

def mask(data,config):

    # create a mask that selects rows where the values in fmg_index columns are greater than 1024
    mask1 = (data[config.fmg_index] > 1024).any(axis=1)

    # create a mask that selects rows where the values in first_position_label_index columns are greater than 2
    mask2 = (data[config.first_positoin_label_inedx] > 3).any(axis=1)

    # combine the masks using the | (or) operator
    mask = mask1 | mask2

    # drop the rows from the DataFrame
    data = data.drop(data[mask].index)


    return data

def is_not_numeric(x):
    try:
        float(x)
        return False
    except ValueError:
        return True
    

def print_not_numeric_vals(df):

    mask = df.drop(['sesion_time_stamp'],axis=1).applymap(is_not_numeric)
    non_numeric_values = df[mask].stack().dropna()
    print(non_numeric_values)

    return non_numeric_values


def make_sequence(config,data: Tensor)->Tensor:

    # Reshape data into sequences of length 20 with 9 features each
    sequence_length = config.sequence_length
    num_features = data.size(1)
    num_samples = data.size(0)
    num_sequences = num_samples // sequence_length

 
    # Reshape the data
    sequenced = data[:num_sequences * sequence_length].view(num_sequences, sequence_length, num_features)



    return sequenced

def create_sliding_sequences(input_tensor, sequence_length):
    sample_size, features = input_tensor.shape
    new_sample_size = sample_size - sequence_length + 1

    sequences = []
    for i in range(new_sample_size):

        sequence = input_tensor[i:i+sequence_length]
        sequences.append(sequence)

    return torch.stack(sequences)

