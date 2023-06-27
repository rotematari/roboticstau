import torch.nn as nn
import torch
from utils import hidden_size_maker



# Define a custom loss function for Root Mean Squared Logarithmic Error (RMSLE)
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        # Compute the RMSLE by taking the square root of the mean squared error between the log-transformed predictions and targets
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    




# Define a fully connected neural network
class fully_connected(nn.Module):
    def __init__(self, config):
        super(fully_connected, self).__init__()

        fully = []
        self.input_size = config.input_size
        # self.relu1 = nn.ReLU()

        self.hidden_size = hidden_size_maker(config)
        assert config.n_layer == len(self.hidden_size) , "size of hidden size list is not equal to n_layer"


        # Create the fully connected layers
        for i in range(config.n_layer):
            if i == 0:
                input_size = config.input_size
            else:
                input_size = self.hidden_size[i-1]

            fully.extend([
                nn.Linear(input_size, self.hidden_size[i],dtype=torch.float64),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size[i],dtype=torch.float64),
                nn.Dropout(config.dropout)
            ])

        self.fully = nn.Sequential(*fully)

        # Create the output layer
        self.out_layer = nn.Linear(self.hidden_size[-1], config.num_labels,dtype=torch.float64)

        # Define several loss functions
        self.mseloss = nn.MSELoss()
        self.huberloss = nn.HuberLoss()
        self.rmsleloss = RMSLELoss()
        
        self.l1loss = nn.L1Loss

    def forward(self, x):
        out = self.fully(x) 
        out = self.out_layer(out)

        return out


    
# if __name__ == '__main__':


    