import time
from os.path import join

import torch
def save_net(checkpoint,args_config ,test_state):

    model_name = 'model_' + time.strftime("%d_%b_%Y_%H:%M", time.gmtime()) + '.pt'
    optimizer_name = 'optimizer_' + time.strftime("%d_%b_%Y_%H:%M", time.gmtime()) + '.pt'
    acc_name = 'accuracy_'+ time.strftime("%d_%b_%Y_%H:%M", time.gmtime()) + '.txt'
    torch.save(checkpoint["state_dict"], join(args_config.model_path, model_name))
    torch.save(checkpoint["optimizer"], join(args_config.model_path, optimizer_name))
    torch.save(test_state,join(args_config.model_path, acc_name))
    print(f"Model's state_dict:{model_name}")
    print(f"optimizer's state_dict:{optimizer_name}")
    print(f'accuracys state dict : {acc_name} ')