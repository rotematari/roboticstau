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



def real_time(model, mean_relaxed, full_arr_std, calibrate=True, s=(200, 6)):
    # try:
    X = real_time_data.Data(mean_relaxed, full_arr_std, calibrate=calibrate, s=s)
    mean_data = X.x.shape
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

def run_test():

    s = (500, 6)
    mean_relaxed = np.zeros(s)
    full_arr_std = 1
    input_size = 12
    calibrate = True
    args_config = parser.parse_args()

    PATH = r'/home/roblab20/Documents/rotem/models/saved_models/model_16_Jan_2023_14:36.pt'
    model = fully_connected.NeuralNet(input_size, args_config)
    model.load_state_dict(torch.load(PATH))
    model.eval()


    run = True
    while run:
        # time.sleep(0.2)
        predicted, calibrate, mean_relaxed, full_arr_std = real_time(model, mean_relaxed, full_arr_std, calibrate, s=s)

        print(predicted)

if __name__ == '__main__':

    run_test()

    # # p1 = multiprocessing.Process(target=plot_serial.run)
    # p2 = multiprocessing.Process(target=run_test)
    #
    # p1.start()
    # p2.start()

