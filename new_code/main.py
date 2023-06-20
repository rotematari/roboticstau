import torch
from torch.optim import Adam 
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

import data.data_proses as data_proses
from models.models import fully_connected as fc 

import utils
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description='Train a neural network.')

# Add arguments for the configuration options
parser.add_argument('--input_size', type=int, default=29, help='The size of the input layer.')
parser.add_argument('--num_labels', type=int, default=12, help='The number of output labels.')
parser.add_argument('--n_layer', type=int, default=3, help='The number of hidden layers.')
parser.add_argument('--hidden_size', nargs='+', type=int, default=[64, 64, 64], help='The size of each hidden layer.')
parser.add_argument('--dropout', nargs='+', type=float, default=[0.1, 0.1, 0.1], help='The dropout rate for each hidden layer.')
# Add arguments for the hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate for the optimizer.')
parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs to train for.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='The weight decay for the optimizer.')
parser.add_argument('--batch_size', type=int, default=100, help='The size of batchs  .')

#dit_path 
parser.add_argument('--data_path', type=str, default='./data', help='The path to the training data.')


# split args 
parser.add_argument('--test_size', type=float, default=0.2, help='The size of the test set.')
parser.add_argument('--val_size', type=float, default=0.2, help='The size of the validation set.')
parser.add_argument('--random_state', type=int, default=42, help='The random state for data splitting.')

if __name__ == '__main__':
    # Parse the command-line arguments
    args = parser.parse_args()

    # Create a configuration object from the parsed arguments
    config = argparse.Namespace(
        input_size=args.input_size,
        num_labels=args.num_labels,
        n_layer=args.n_layer,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        batch_size = args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        data_path=args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    # Create an instance of the FullyConnected class using the configuration object
    net = fc(config)

    
    # load data 
    data = data_proses.data_loder(config=config)
    fmg_df, _, label_df = data_proses.sepatare_data(data)

    # subtract bias 
    fmg_df = data_proses.find_bias


    # std norm
    # Split the data into training and test sets
    train_fmg, test_fmg, train_label, test_label = train_test_split(
        fmg_df, label_df, test_size=config.test_size, random_state=config.random_state)

    # Split the training data into training and validation sets
    train_fmg, val_fmg, train_label, val_label = train_test_split(
        train_fmg, train_label, test_size=config.val_size / (1 - config.test_size), random_state=config.random_state)




    # Create TensorDatasets for the training, validation, and test sets
    train_dataset = TensorDataset(train_fmg, train_label)
    val_dataset = TensorDataset(val_fmg, val_label)
    test_dataset = TensorDataset(test_fmg, test_label)

    # Create DataLoaders for the training, validation, and test sets
  
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)


 







    train_losses, val_losses = utils.train(config=config,train_loader=train_loader
                                                                             ,val_loader=val_loader,net=net )
    








