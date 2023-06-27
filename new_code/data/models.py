import torch.nn as nn
import torch

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    

class fully_conected(nn.Module):
    def __init__(self,args_config):
        super(fully_conected, self).__init__()

        fully = []
        self.input_size = args_config.input_size
        # self.relu1 = nn.ReLU()

        for i in range(args_config.n_layer):
            if i==0:
                input_size = args_config.input_size
            else :
                input_size = args_config.hidden_size[i-1]

            fully.extend[
                nn.Linear(input_size, args_config.hidden_size[i],dtype=torch.float64),
                nn.ReLU(),
                nn.BatchNorm1d(args_config.hidden_size[i]),
                nn.Dropout1d(args_config.dropout[i])
            ]

        self.fully = nn.Sequential(*fully)

        self.out_layer = nn.Linear(args_config.hidden_size[-1], args_config.num_labels,dtype=torch.float64)
        self.mseloss = nn.MSELoss()
        self.huberloss = nn.HuberLoss()
        self.rmsleloss = RMSLELoss()
        self.l1loss = nn.L1Loss

    def forward(self, x):
        out = self.fully(x) 
        out = self.out_layer(out)

        return out
    
