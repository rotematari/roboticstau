import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import join

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import MinMaxScaler

import wandb
import time
from torch.optim.lr_scheduler import LambdaLR

# TODO: add time for iteration for the eval model 
def train(config, train_loader, val_loader,model,device='cpu',wandb_run=None):
    criterion = MSELoss()
    
    if model.name == "TransformerModel" or model.name == "TemporalConvNet":
        # Learning rate warm-up
        warmup_steps = 4000
        initial_lr = 1e-4  # Starting learning rate

        # Custom lambda for learning rate schedule
        lr_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)

        optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=config.weight_decay)

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
                if outputs.dim() == 3:
                    loss = criterion(outputs[:,-1:,:].squeeze(), targets[:,-1:,:].squeeze())
                else:
                    loss = criterion(outputs, targets[:,-1,:])
            else:
                loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            if model.name == "TransformerModel" or model.name == "TemporalConvNet":
                scheduler.step()  # Update the learning rate

            # Update the epoch loss and accuracy
            train_loss += loss.item()
           

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        val_loss = 0
        total_time = 0
        # Evaluate on the validation set
        with torch.no_grad():
            model.eval()
            
            for i,(inputs, targets) in enumerate(val_loader):
                
                start_time = time.time()
                inputs = inputs.to(device=device)
                targets = targets.to(device=device)

                outputs = model(inputs)
                if config.sequence:
                    if outputs.dim() == 3:
                        v_loss = criterion(outputs[:,-1:,:].squeeze(), targets[:,-1:,:].squeeze())
                    else:
                        v_loss = criterion(outputs, targets[:,-1,:])
                else:
                    v_loss = criterion(outputs, targets)

                val_loss += v_loss.item()
                end_time = time.time()
                total_time += (end_time - start_time)

            
            val_loss /= len(val_loader)
            avg_iter_time = total_time/len(val_loader)
            
        # Save the validation loss 
        val_losses.append(val_loss)
        
        # Print the epoch loss and accuracy values
        print(f'Epoch: {epoch} Train Loss: {train_loss}  Val Loss: {val_loss} time for one iteration {1000*avg_iter_time:.4f} ms')
        if wandb_run is not None:
            # log metrics to wandb
            wandb_run.log({"Train Loss": train_loss, "Val_loss": val_loss})

        if(best_val_loss < val_loss):
            time_stamp = time.strftime("%d_%m", time.gmtime())
            best_val_loss = val_loss
            filename = model.name + '_epoch_' +str(epoch+1)+'_date_'+time_stamp + '.pt'
            checkpoint_path = join(config.model_path,filename)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'config': config,
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

def plot_results(config,preds,targets,wandb_run=None):

    # Create a figure and a grid of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(10, 4),sharex=True,sharey=True)  # Adjust figsize as needed

    # Plot data on the first subplot
    axes[0,0].plot(preds[:,:9])
    axes[0,0].set_title('Plot of preds location')
    axes[0,0].grid()


    axes[0,1].plot(preds[:,9:])
    axes[0,1].set_title('Plot of preds V ')
    axes[0,1].grid()

    # Plot data on the second subplot
    axes[1,0].plot(targets[:,:9])
    axes[1,0].set_title('Plot of targets location')
    axes[1,0].grid()
    if config.with_velocity:
        axes[1,1].plot(targets[:,9:])
        axes[1,1].set_title('Plot of targets V')
        axes[1,1].grid()
        

    # Adjust layout to prevent overlap
    plt.tight_layout()


    if wandb_run is not None:
        # Log the figure to wandb
        wandb.log({"preds and targets": plt})
    else:
        # Show the plots
        plt.show()

def model_eval_metric(config,model,test_loader,
                    #   label_max_val,label_min_val ,
                    data_processor,
                    device='cpu',wandb_run=None):
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
            outputs = model(inputs)
            if outputs.dim() == 3:
                outputs = outputs[:,-1,:]
            size = outputs.size(0)
            
            if config.norm_labels:
                outputs = data_processor.label_scaler.inverse_transform(outputs)
                targets = data_processor.label_scaler.inverse_transform(targets)
                # outputs = min_max_unnormalize(outputs.detach().cpu().numpy(),np.tile(label_min_val,(size,1)),np.tile(label_max_val,(size,1)))
                # targets = min_max_unnormalize(targets.detach().cpu().numpy(),np.tile(label_min_val,(targets.size(0),targets.size(1),1)),np.tile(label_max_val,(targets.size(0),targets.size(1),1)))
            
            if config.plot_pred:
                plot_results(config,outputs[100:2000].cpu().detach().numpy(),targets[100:2000,-1,:].cpu().detach().numpy(),wandb_run=wandb_run)

            dist = np.sqrt(((outputs.cpu().detach().numpy() - targets[:,-1:,:].view(-1,config.num_labels).cpu().detach().numpy())**2).sum(axis=0)/size)
        else:
            outputs = model(inputs[0:2000])
            size = outputs.size(0)
            if config.norm_labels:
                outputs = data_processor.label_scaler.inverse_transform(outputs)
                targets = data_processor.label_scaler.inverse_transform(targets)
                # outputs = min_max_unnormalize(outputs.detach().cpu().numpy(),np.tile(label_min_val,(size,1)),np.tile(label_max_val,(size,1)))
                # targets = min_max_unnormalize(targets.detach().cpu().numpy(),np.tile(label_min_val,(targets.size(0),1)),np.tile(label_max_val,(targets.size(0),1)))

            if config.plot_pred:
                plot_results(config,outputs[0:2000].cpu().detach().numpy(),targets[0:2000,:].view(-1,config.num_labels).cpu().detach().numpy(),wandb_run=wandb_run)

            dist = np.sqrt(((outputs.cpu().detach().numpy() - targets[:size].cpu().detach().numpy())**2).sum(axis=0)/size)

    return dist

# depricated
def min_max_unnormalize(data, min_val, max_val,bottom=-1, top=1):

    return torch.tensor(((data-bottom)/(top-bottom)) * (max_val - min_val) + min_val)

#depricated 
def min_max_normalize(data,bottom=-1, top=1):

    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    norm = bottom+(data - min_val) / (max_val - min_val)*(top-bottom)

    return norm,max_val,min_val

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

def data_loader(config):
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

def get_label_axis(labels,config):
    #label_inedx = ['M1x','M1y','M1z','M2x','M2y','M2z','M3x','M3y','M3z','M4x','M4y','M4z']
    # Create a copy of the labels DataFrame slice
    # labels_copy = labels[config.first_positoin_label_inedx].copy()
    labels = labels.copy()
    # Now perform the operations on the copy
    labels.loc[:,['M1x','M2x','M3x','M4x']]  = labels[['M1x','M2x','M3x','M4x']].sub(labels['M1x'], axis=0)
    labels.loc[:,['M1y','M2y','M3y','M4y']] = labels[['M1y','M2y','M3y','M4y']].sub(labels['M1y'], axis=0)
    labels.loc[:,['M1z','M2z','M3z','M4z']] = labels[['M1z','M2z','M3z','M4z']].sub(labels['M1z'], axis=0)
   
    return labels[config.position_label_index]

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
    return label_df

def mask(data,config):

    # create a mask that selects rows where the values in fmg_index columns are greater than 1024
    mask1 = (data[config.fmg_index] > 1024).any(axis=1)

    # create a mask that selects rows where the values in first_position_label_index columns are greater than 2
    mask2 = (data[config.first_position_label_index] > 3).any(axis=1)

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

def create_sliding_sequences(input_tensor, sequence_length):
    sample_size, features = input_tensor.shape
    new_sample_size = sample_size - sequence_length + 1

    sequences = []
    for i in range(new_sample_size):

        sequence = input_tensor[i:i+sequence_length]
        sequences.append(sequence)

    return torch.stack(sequences)
