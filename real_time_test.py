import argparse

import numpy as np

import real_time_data
from models import fully_connected
import torch

parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# hyper meters
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 20)')
parser.add_argument('--epoch', type=int, default=8,
                    help='input num of epoch (default: 5')
parser.add_argument('--num_classes', type=int, default=4,
                    help='input num of classes (default: 4')
parser.add_argument('-lr', '--learning_rate', type=int, default= 0.07377084647975934,
                    help='input learning rate (default: 0.001')
parser.add_argument('-wd', '--weight_decay', type=int, default=2.643887676317213e-05,
                    help='input weight_decay (default: 0.0001')
parser.add_argument('--hidden_size_1', type=int, default=64,
                    help='input hidden_size_1 (default: 5')
parser.add_argument('--hidden_size_2', type=int, default=128,
                    help='input hidden_size_2 (default: 5')
parser.add_argument('--hidden_size_3', type=int, default=64,
                    help='input hidden_size_3 (default: 5')
parser.add_argument('--dropout_1', type=int, default=0.025321265231112014,

                    help='input dropout_1(default: 0.1')
parser.add_argument('--dropout_2', type=int, default= 0.03193992930059613,
                    help='input dropout_2 (default: 0.1')
parser.add_argument('--dropout_3', type=int, default=0.021450112872249766,
                    help='input dropout_3(default: 0.1')

def real_time(model, mean_relaxed, full_arr_std, calibrate=True, s=(200, 6)):

    X = real_time_data.Data(mean_relaxed, full_arr_std, calibrate=calibrate, s=s)

    mean_relaxed = X.mean_relaxed
    full_arr_std = X.full_arr_std
    calibrate = False
    outputs = model(X.x)
    # print(outputs.data)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)

    return predicted, calibrate, mean_relaxed, full_arr_std


if __name__ == '__main__':
    s = (100, 6)
    mean_relaxed = np.zeros(s)
    full_arr_std = 1
    input_size = 6
    calibrate = True
    args_config = parser.parse_args()
    PATH = r'/home/roblab20/Documents/rotem/models/saved_models/model_10_Jan_2023_16:08.pt'
    model = fully_connected.NeuralNet(input_size, args_config)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    run = True
    while run:
        predicted, calibrate, mean_relaxed, full_arr_std = real_time(model, mean_relaxed, full_arr_std, calibrate, s=s)
        print(predicted)
