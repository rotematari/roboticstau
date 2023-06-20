import torch
from torch.optim import Adam 


import data.data_proses as data_proses
import models.fully_connected as fc 
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
parser.add_argument('--data_path', type=str, default='./data', help='The path to the training data.')





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
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        data_path=args.data_path
    )
    # Create an instance of the FullyConnected class using the configuration object
    net = fc(config)





