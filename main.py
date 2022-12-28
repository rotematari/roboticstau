import argparse
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

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="pointing",
    config={}

)

parser = argparse.ArgumentParser(description='Training Config', add_help=False)


#

parser.add_argument('--initialize_weights', type=str, default=True
                    , help='initialize weights')
parser.add_argument('--pre_train_own_model', type=str, default=False,
                    help='load pretrained model weights')

# hyper meters
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 20)')
parser.add_argument('--epoch', type=int, default=5,
                    help='input num of epoch (default: 5')
parser.add_argument('--num_classes', type=int, default=4,
                    help='input num of classes (default: 4')
parser.add_argument('-lr', '--learning_rate', type=int, default=0.001,
                    help='input learning rate (default: 0.001')
parser.add_argument('-wd', '--weight_decay', type=int, default=0.0001,
                    help='input weight_decay (default: 0.0001')
parser.add_argument('--hidden_size_1', type=int, default=10,
                    help='input hidden_size_1 (default: 5')
parser.add_argument('--hidden_size_2', type=int, default=10,
                    help='input hidden_size_2 (default: 5')
parser.add_argument('--hidden_size_3', type=int, default=10,
                    help='input hidden_size_3 (default: 5')
parser.add_argument('--dropout_1', type=int, default=0.1,
                    help='input dropout_1(default: 0.1')
parser.add_argument('--dropout_2', type=int, default=0.1,
                    help='input dropout_2 (default: 0.1')
parser.add_argument('--dropout_3', type=int, default=0.1,
                    help='input dropout_3(default: 0.1')


def main(args_config,device):
    # data loader
    train_data = data_loader.Data(train=True, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)
    # test_data = data_loader.Data(train=False, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)

    input_size = train_data.n_featurs

    # split train
    test_abs = int(len(train_data) * 0.8)
    train_subset, validation_subset = random_split(train_data, [test_abs, len(train_data) - test_abs])

    # torch data loader
    ## train

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=args_config.batch_size, shuffle=True,
                                               drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_subset, batch_size=args_config.batch_size,
                                                    shuffle=True, drop_last=True)

    # ##test
    # test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args_config.batch_size, shuffle=False,
    #                                           drop_last=True)

    model = fully_connected.NeuralNet(train_data.n_featurs, args_config)

    if args_config.pre_train_own_model:
        model_weights_path = r'/home/roblab15/Documents/FMG_project/models/'
        model.load_state_dict(model_weights_path)
    elif args_config.initialize_weights :
        model.apply(initializer.initialize_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args_config.learning_rate,
                                     weight_decay=args_config.weight_decay)

    for epoch in range(0, args_config.epoch):
        model, optimizer = Run_model.train_epoch(epoch=epoch, model=model, train_loader=train_loader
                                                 ,criterion=criterion,optimizer=optimizer,
                                                 args_config=args_config,device=device)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    state = check_accuracy(state=state,
                           model=model,
                           optimizer=optimizer,
                           validation_loader=validation_loader,
                           args_config=args_config,
                           scheduler=scheduler, device=device,
                           epochs=epoch, IOU_metrics=IOU_metrics)

if __name__ == '__main__':
    args_config = parser.parse_args()
    # adds all of the arguments as config variables
    wandb.config.update(args_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args_config, device)
