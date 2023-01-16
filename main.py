import argparse
import time
from os.path import join

import wandb

import torch
import torch.utils.data
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.init as init

import paramaters
import data_loader
from models import fully_connected, initializer
import Run_model
import utils

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="pointing",
    config={}

)

parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# destanations

parser.add_argument('--model_path', type=str, default=r'/home/roblab20/Documents/rotem/models/saved_models'
                    , help='enter model dir path')

parser.add_argument('--initialize_weights', type=str, default=True
                    , help='initialize weights')
parser.add_argument('--pre_train_own_model', type=str, default=False,
                    help='if pretrained model weights :True')
parser.add_argument('--train_model', type=str, default=True,
                    help='to train model:True')

# hyper meters
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 20)')
parser.add_argument('--epoch', type=int, default=50,
                    help='input num of epoch (default: 5')
parser.add_argument('--num_classes', type=int, default=4,
                    help='input num of classes (default: 4')
parser.add_argument('-lr', '--learning_rate', type=int, default= 0.01306988530510535,
                    help='input learning rate (default: 0.001')
parser.add_argument('-wd', '--weight_decay', type=int, default=1.3972998390212782e-06,
                    help='input weight_decay (default: 0.0001')
parser.add_argument('--hidden_size_1', type=int, default=4,
                    help='input hidden_size_1 (default: 5')
parser.add_argument('--hidden_size_2', type=int, default=64,
                    help='input hidden_size_2 (default: 5')
parser.add_argument('--hidden_size_3', type=int, default=64,
                    help='input hidden_size_3 (default: 5')
parser.add_argument('--dropout_1', type=int, default=0.025815114547734695,
                    help='input dropout_1(default: 0.1')
parser.add_argument('--dropout_2', type=int, default= 0.011199123124155278,
                    help='input dropout_2 (default: 0.1')
parser.add_argument('--dropout_3', type=int, default=0.048169880128418226,
                    help='input dropout_3(default: 0.1')


def main(args_config, device):
    # data loader
    train_data = data_loader.Data(train=True, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)
    test_data = data_loader.Data(train=False, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)

    input_size = train_data.n_featurs

    # split train
    test_abs = int(len(train_data) * 0.9)
    train_subset, validation_subset = random_split(train_data, [test_abs, len(train_data) - test_abs])

    # torch data loader
    ## train

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=args_config.batch_size, shuffle=True,
                                               drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_subset, batch_size=args_config.batch_size,
                                                    shuffle=True, drop_last=True)

    # ##test

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args_config.batch_size, shuffle=True,
                                              drop_last=True)

    model = fully_connected.NeuralNet(train_data.n_featurs, args_config)

    if args_config.pre_train_own_model:
        model_weights_path = r'/home/roblab20/Documents/rotem/models/saved_models/model_10_Jan_2023_16:08.pt'
        model.load_state_dict(torch.load(model_weights_path))
    elif args_config.initialize_weights:
        model.apply(initializer.initialize_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args_config.learning_rate,
                                 weight_decay=args_config.weight_decay)

    model.to(device)

    if args_config.train_model:
        for epoch in range(0, args_config.epoch):
            model, optimizer = Run_model.train_epoch(epoch=epoch, model=model, train_loader=train_loader
                                                     , criterion=criterion, optimizer=optimizer,
                                                     args_config=args_config, device=device)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),

        }

    model.eval()
    state = Run_model.check_accuracy(model=model,
                                     optimizer=optimizer,
                                     validation_loader=validation_loader,
                                     args_config=args_config,
                                     device=device,
                                     criterion=criterion, )

    print("test")
    model.eval()
    test_state = Run_model.check_accuracy(model=model,
                                          optimizer=optimizer,
                                          validation_loader=test_loader,
                                          args_config=args_config,
                                          device=device,
                                          criterion=criterion, )

    # saves model and optimizer
    save = 0
    # # save = input(" to save net press 1 ")
    if test_state > 0.9:
        utils.save_net(checkpoint, args_config, test_state)


if __name__ == '__main__':
    args_config = parser.parse_args()
    # adds all of the arguments as config variables
    wandb.config.update(args_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(10):
        main(args_config, device)
