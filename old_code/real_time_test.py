import argparse

import numpy as np
from matplotlib import pyplot as plt, animation
import time
import real_time_data
from models import fully_connected
import torch
import multiprocessing
from plots import plot_serial

parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# hyper meters
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 20)')
parser.add_argument('--epoch', type=int, default=12,
                    help='input num of epoch (default: 5')
parser.add_argument('--num_classes', type=int, default=4,
                    help='input num of classes (default: 4')
parser.add_argument('-lr', '--learning_rate', type=int, default=0.01786545311029036,
                    help='input learning rate (default: 0.001')
parser.add_argument('-wd', '--weight_decay', type=int, default=2.6855359747729944e-05,
                    help='input weight_decay (default: 0.0001')
parser.add_argument('--hidden_size_1', type=int, default=64,
                    help='input hidden_size_1 (default: 5')
parser.add_argument('--hidden_size_2', type=int, default=128,
                    help='input hidden_size_2 (default: 5')
parser.add_argument('--hidden_size_3', type=int, default=32,
                    help='input hidden_size_3 (default: 5')
parser.add_argument('--dropout_1', type=int, default=0.027793683687556105,
                    help='input dropout_1(default: 0.1')
parser.add_argument('--dropout_2', type=int, default=0.04720404224908827,
                    help='input dropout_2 (default: 0.1')
parser.add_argument('--dropout_3', type=int, default=0.039895370370821547,
                    help='input dropout_3(default: 0.1')
parser.add_argument('--sensors', type=list,
                    default=['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S10'],
                    help='sensors to input(default: 0.1')
parser.add_argument('--std', type=list,
                    default=['S1_std', 'S2_std', 'S3_std', 'S4_std', 'S5_std', 'S7_std', 'S11_std'],
                    help='sensors to input(default: 0.1')
parser.add_argument('--all_data', type=list,
                    default=['time', 'Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz', 'S1', 'S2', 'S3', 'S4', 'S5',
                             'S6', 'S7', 'S8', 'S9', 'S10', 'S11'],
                    help='sensors to input(default: 0.1')


def real_time(model, args_config, mean_relaxed, full_arr_std, calibrate=True, s=(200, 6)):
    # try:
    X = real_time_data.Data(mean_relaxed, full_arr_std, args_config, calibrate=calibrate, s=s)

    mean_relaxed = X.mean_relaxed
    full_arr_std = X.full_arr_std
    calibrate = False
    outputs = model(X.x)
    # print(outputs.data)
    # max returns (value ,index)

    _, predicted = torch.max(outputs.data, 1)

    # except:
    #     print('got none')
    #     predicted = 1

    return predicted, calibrate, mean_relaxed, full_arr_std


def run_test(args_config):
    s = (500, len(args_config.sensors))
    mean_relaxed = np.zeros(s)
    full_arr_std = 1
    input_size = 14
    calibrate = True

    PATH = r'/home/robotics20/Documents/rotem/models/model_25_Jan_2023_11:34.pt'
    model = fully_connected.NeuralNet(input_size, args_config)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    run = True
    while run:
        # time.sleep(0.2)
        predicted, calibrate, mean_relaxed, full_arr_std = real_time(model, args_config, mean_relaxed, full_arr_std,
                                                                     calibrate, s=s)

        print(predicted)


if __name__ == '__main__':
    args_config = parser.parse_args()
    run_test(args_config)
    # # p1 = multiprocessing.Process(target=plot_serial.run)
    # p2 = multiprocessing.Process(target=run_test)
    #
    # p1.start()
    # p2.start()
