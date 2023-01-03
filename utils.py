import time
from os.path import join

import torch
def save_net(checkpoint,args_config):

    model_name = 'model_' + time.strftime("%d_%b_%Y_%H:%M", time.gmtime()) + '.pt'
    optimizer_name = 'optimizer_' + time.strftime("%d_%b_%Y_%H:%M", time.gmtime()) + '.pt'

    torch.save(checkpoint["state_dict"], join(args_config.model_path, model_name))
    torch.save(checkpoint["optimizer"], join(args_config.model_path, optimizer_name))
    print(f"Model's state_dict:{model_name}")
    print(f"optimizer's state_dict:{optimizer_name}")