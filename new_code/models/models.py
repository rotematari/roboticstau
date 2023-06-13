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

        # self.dropout1 = nn.Dropout1d(args_config.dropout_1)
        # self.dropout2 = nn.Dropout1d(args_config.dropout_2)
        # self.dropout3 = nn.Dropout1d(args_config.dropout_3)

        # self.batch_norm_1 = nn.BatchNorm1d(args_config.hidden_size_1)
        # self.batch_norm_2 = nn.BatchNorm1d(args_config.hidden_size_2)
        # self.batch_norm_3 = nn.BatchNorm1d(args_config.hidden_size_3)
        
        # self.l1 = nn.Linear(input_size, args_config.hidden_size_1)
        # self.l2 = nn.Linear(args_config.hidden_size_1, args_config.hidden_size_2)
        # self.l3 = nn.Linear(args_config.hidden_size_2, args_config.hidden_size_3)
        # self.l4 = nn.Linear(args_config.hidden_size_3, args_config.num_classes)

        self.relu1 = nn.ReLU()

        for i in range(args_config.n_layer):
            if i==0:
                input_size = args_config.input_size
            else :
                input_size = args_config.hidden_size[i-1]

            fully.extend[
                nn.Linear(input_size, args_config.hidden_size[i]),
                nn.ReLU(),
                nn.BatchNorm1d(args_config.hidden_size[i]),
                nn.Dropout1d(args_config.dropout[i])
            ]

        self.fully = nn.Sequential(*fully)

        self.out_layer = nn.Linear(args_config.hidden_size[-1], args_config.num_labels)
        self.mseloss = nn.MSELoss()
        self.huberloss = nn.HuberLoss()
        self.rmsleloss = RMSLELoss()
        
        self.l1loss = nn.L1Loss

    def forward(self, x):
        out = self.fully(x) 
        out = self.out_layer(out)
        # # layer 1
        # out = self.relu1(self.l1(x))
        # out = self.batch_norm_1(out)
        # out = self.dropout1(out)
        # # layer 2
        # out = self.relu2(self.l2(out))
        # out = self.batch_norm_2(out)
        # out = self.dropout2(out)
        # # layer 3
        # out = self.relu3(self.l3(out))
        # out = self.batch_norm_3(out)
        # out = self.dropout3(out)
        # # outer layer
        # out = self.l4(out)
        return out
    
