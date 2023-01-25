# sphinx_gallery_thumbnail_number = 2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import data_loader
import data_agmuntation
import paramaters

parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# destanations

parser.add_argument('--model_path', type=str, default=r'/home/robotics20/Documents/rotem/models'
                    , help='enter model dir path')
parser.add_argument('--data_path', type=str, default=r'/home/robotics20/Documents/rotem/data'
                    , help='enter data dir path')
parser.add_argument('--sensors', type=list,
                    default=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11'],
                    help='sensors to input(default: 0.1')

sample_rate = 10
def main(args_config):
    items = args_config.sensors
    x = data_loader.Data(args_config,train=True, dirpath=args_config.data_path, items=args_config.sensors)



    y1 = x.X[:, 0].numpy()
    y2 = x.X[:, 1].numpy()
    y3 = x.X[:, 2].numpy()
    y4 = x.X[:, 3].numpy()
    y5 = x.X[:, 4].numpy()
    y6 = x.X[:, 5].numpy()
    y7 = x.X[:, 6].numpy()
    y8 = x.X[:, 7].numpy()
    y9 = x.X[:, 8].numpy()
    y10 = x.X[:, 9].numpy()
    y11 = x.X[:, 10].numpy()

    lables = x.Y

    # list(y)
    plt.figure(1)
    plt.plot(list(y1))
    plt.plot(list(y2))
    plt.plot(list(y3))
    plt.plot(list(y4))
    plt.plot(list(y5))
    plt.plot(list(y6))
    plt.plot(list(y7))
    plt.plot(list(y8),'c')
    plt.plot(list(y9),'r')
    plt.plot(list(y10),'g')
    plt.plot(list(y11),'b')
    plt.plot(list(lables))
    plt.legend(items)


    # plt.figure(2)
    # z = data_loader.Data(train=False, dirpath=dirpath, items=items)
    #
    #
    #
    # y1 = z.X[:, 0].numpy()
    # y2 = z.X[:, 1].numpy()
    # y3 = z.X[:, 2].numpy()
    # # y4 = z.X[:, 3].numpy()
    # # y5 = z.X[:, 4].numpy()
    # # y6 = z.X[:, 5].numpy()
    # lables = z.Y
    #
    # # list(y)
    # plt.plot(list(y1))
    # plt.plot(list(y2))
    # plt.plot(list(y3))
    # # plt.plot(list(y4))
    # # plt.plot(list(y5))
    # # plt.plot(list(y6))
    # plt.plot(list(lables))
    # plt.legend(items)
    plt.show()

if __name__ == '__main__':

    args_config = parser.parse_args()

    main(args_config)