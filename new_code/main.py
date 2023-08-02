import torch
from torch.optim import Adam 
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
import sys
import os
import pandas as pd 

PATH = os.path.join(os.path.dirname(__file__),"/models")
sys.path.insert(0,PATH)


import data.data_proses as data_proses
from models.models import fully_connected as fc 

import utils
import argparse
import wandb

import yaml

with open('/home/robotics20/Documents/rotem/new_code/config.yaml', 'r') as f:
    args = yaml.safe_load(f)

config = argparse.Namespace(**args)

def init(): 

    global wandb_on 
    wandb_on = 1 # (1 = True, 0 = False)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    return


# Define the argument parser
parser = argparse.ArgumentParser(description='Train a neural network.')

# Add arguments for the configuration options
parser.add_argument('--input_size', type=int, default=config.input_size, help='The size of the input layer.')
parser.add_argument('--num_labels', type=int, default=config.num_labels, help='The number of output labels.')
parser.add_argument('--n_layer', type=int, default=config.n_layer, help='The number of hidden layers.')
parser.add_argument('--hidden_size', nargs='+', type=int, default=config.hidden_size, help='The size of each hidden layer.')
parser.add_argument('--dropout', nargs='+', type=float, default=config.dropout, help='The dropout rate for each hidden layer.')
parser.add_argument('--multiplier', nargs='+', type=float, default=config.multiplier, help='The multiplier gor hidden state.')
# Add arguments for the hyperparameters
parser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='The learning rate for the optimizer.')
parser.add_argument('--num_epochs', type=int, default=config.num_epochs, help='The number of epochs to train for.')
parser.add_argument('--weight_decay', type=float, default=config.weight_decay, help='The weight decay for the optimizer.')
parser.add_argument('--batch_size', type=int, default=config.batch_size, help='The size of batchs  .')

#dit_path 
parser.add_argument('--data_path', type=str, default=config.data_path, help='The path to the training data.')


# split args 
parser.add_argument('--test_size', type=float, default=config.test_size, help='The size of the test set.')
parser.add_argument('--val_size', type=float, default=config.test_size, help='The size of the validation set.')
parser.add_argument('--random_state', type=int, default=config.random_state, help='The random state for data splitting.')

if __name__ == '__main__':

    init()
    # wandb 
    # start a new wandb run to track this script
    wandb.init(project="armModeling",entity='fmgrobotics',config=config)
    config = wandb.config
    print(config)


    # Parse the command-line arguments
    args = parser.parse_args()

    #update wandb 
    wandb.config.update(args)

    # Create an instance of the FullyConnected class using the configuration object
    net = fc(config)
    net= net.to(device=device)

    
    # load data 
    data = data_proses.data_loder(config=config)
    data = data[config.fmg_index+config.first_positoin_label_inedx+config.sesion_time_stamp].dropna()


    # drop bad data 
    data = data_proses.mask(data,config)

    # separate
    fmg_df, _, label_df = data_proses.sepatare_data(data,config=config,first=True)

    
    
    # find zero axis 
    label_df = data_proses.get_label_axis(label_df,config=config)

    # subtract bias 
    fmg_df = data_proses.subtract_bias(fmg_df)


    # std norm
    fmg_df = data_proses.std_division(fmg_df)

    # TODO: add here agmuntations 
    # add velocity 
    label_df = data_proses.calc_velocity(config,label_df)


    #normalize labels 
    label_df = utils.normalize(label_df)

    data = pd.concat([fmg_df,label_df],axis=1).drop_duplicates().reset_index(drop=True).dropna()
    fmg_df,_,label_df = data_proses.sepatare_data(data,config=config,first=False)
    #drop time stamp 
    fmg_df = fmg_df.drop('sesion_time_stamp', axis=1)
    label_df = label_df.drop('sesion_time_stamp', axis=1)


    # Split the data into training and test sets
    train_fmg, test_fmg, train_label, test_label = train_test_split(
        fmg_df, label_df, test_size=config.test_size, random_state=config.random_state)

    # Split the training data into training and validation sets
    train_fmg, val_fmg, train_label, val_label = train_test_split(
        train_fmg, train_label, test_size=config.val_size / (1 - config.test_size), random_state=config.random_state)




    # Create TensorDatasets for the training, validation, and test sets
    train_dataset = TensorDataset(torch.tensor(train_fmg.to_numpy()), torch.tensor(train_label.to_numpy()))
    val_dataset = TensorDataset(torch.tensor(val_fmg.to_numpy()), torch.tensor(val_label.to_numpy()))
    test_dataset = TensorDataset(torch.tensor(test_fmg.to_numpy()), torch.tensor(test_label.to_numpy()))


    # Create DataLoaders for the training, validation, and test sets
  
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,drop_last=True)

    # train 
    train_losses, val_losses = utils.train(config=config,train_loader=train_loader
                                                                             ,val_loader=val_loader,net=net,device=device )
    


    #test
    eval = utils.model_eval_metric(config,net,test_loader,device=device)
    test_loss = utils.test(net=net,config=config,test_loader=test_loader,device=device)
    print(f'eval_metric {eval}')

    if wandb_on:
        wandb.log({'eval_metric':eval,'Test_loss':test_loss})

    #plot train
    utils.plot_losses(train_losses, val_losses=val_losses)




    if wandb_on:
        wandb.finish()


    



    








