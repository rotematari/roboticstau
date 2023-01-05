import argparse

import numpy as np

import real_time_data
from models import fully_connected
import torch





parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# hyper meters
parser.add_argument('--batch_size', type=int, default=30,
                    help='input batch size for training (default: 20)')
parser.add_argument('--epoch', type=int, default=21,
                    help='input num of epoch (default: 5')
parser.add_argument('--num_classes', type=int, default=4,
                    help='input num of classes (default: 4')
parser.add_argument('-lr', '--learning_rate', type=int, default=0.0006042626727762781,
                    help='input learning rate (default: 0.001')
parser.add_argument('-wd', '--weight_decay', type=int, default=3.643379907226691e-05,
                    help='input weight_decay (default: 0.0001')
parser.add_argument('--hidden_size_1', type=int, default=16,
                    help='input hidden_size_1 (default: 5')
parser.add_argument('--hidden_size_2', type=int, default=16,
                    help='input hidden_size_2 (default: 5')
parser.add_argument('--hidden_size_3', type=int, default=16,
                    help='input hidden_size_3 (default: 5')
parser.add_argument('--dropout_1', type=int, default=0.015376617828307836,
                    help='input dropout_1(default: 0.1')
parser.add_argument('--dropout_2', type=int, default=0.14942078024203984,
                    help='input dropout_2 (default: 0.1')
parser.add_argument('--dropout_3', type=int, default=0.03077733834382956,
                    help='input dropout_3(default: 0.1')
def real_time(model ,mean_relaxed, calibrate=True ,s=(200, 6)):



    X = real_time_data.Data(mean_relaxed, calibrate=calibrate,s=s)

    mean_relaxed =X.mean_relaxed
    calibrate = False
    outputs = model(X.x)
    # print(outputs.data)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)

    return predicted, calibrate, mean_relaxed


if __name__ == '__main__':
    s = (200, 6)
    mean_relaxed = np.zeros(s)
    input_size = 6
    calibrate =True
    args_config = parser.parse_args()
    PATH = r'/home/roblab20/Documents/rotem/models/saved_models/model_05_Jan_2023_15:56.pt'
    model = fully_connected.NeuralNet(input_size, args_config)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    run = True
    while run:

        predicted, calibrate, mean_relaxed = real_time(model, mean_relaxed, calibrate, s=s)
        print(predicted)

