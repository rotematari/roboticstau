import torch.nn as nn
import torch
from utils import hidden_size_maker

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    

class fully_conected(nn.Module):
    def __init__(self,config):
        super(fully_conected, self).__init__()

        fully = []
        self.input_size = config.input_size

        
        # self.relu1 = nn.ReLU()
        self.hidden_size = hidden_size_maker(config)
        assert config.n_layer == len(self.hidden_size) , "size of hidden size list is not equal to n_layer"


        for i in range(config.n_layer):
            if i==0:
                input_size = self.input_size
            else :
                input_size = self.hidden_size[i-1]

            fully.extend[
                nn.Linear(input_size, self.hidden_size[i],dtype=torch.float64),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size[i]),
                nn.Dropout1d(config.dropout[i])
            ]

        self.fully = nn.Sequential(*fully)

        self.out_layer = nn.Linear(self.hidden_size[-1], config.num_labels,dtype=torch.float64)
        self.mseloss = nn.MSELoss()
        self.huberloss = nn.HuberLoss()
        self.rmsleloss = RMSLELoss()
        self.l1loss = nn.L1Loss

    def forward(self, x):
        out = self.fully(x) 
        out = self.out_layer(out)

        return out
    
