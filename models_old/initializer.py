
import torch.nn as nn
import torch.nn.init as init
# https://pytorch.org/docs/stable/nn.init.html


def initialize_weights(model):

    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data, gain=init.calculate_gain('relu'))
        nn.init.constant_(model.bias.data, 0)

    elif isinstance(model, nn.BatchNorm1d):
        nn.init.constant_(model.bias.data, 1)
        nn.init.constant_(model.bias.data, 0)
