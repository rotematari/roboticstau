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
    def __init__(self,config,seq=True):
        super(fully_conected, self).__init__()

        fully = []
        
        if seq:
            self.input_size = config.seq_size
            self.output_size = config.label_seq_size
        else :
            self.input_size = config.input_size

        
       
        self.hidden_size = hidden_size_maker(config)
        assert config.n_layer == len(self.hidden_size) , "size of hidden size list is not equal to n_layer"


        for i in range(config.n_layer):
            if i==0:
                input_size = self.input_size
            else :
                input_size = self.hidden_size[i-1]

            fully.extend([
                nn.Linear(input_size, self.hidden_size[i],dtype=torch.float64),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size[i],dtype=torch.float64),
                nn.Dropout1d(config.dropout),
               
            ])

        self.fully = nn.Sequential(*fully)

        self.out_layer = nn.Linear(self.hidden_size[-1], self.output_size ,dtype=torch.float64)
        self.mseloss = nn.MSELoss()
        self.huberloss = nn.HuberLoss()
        self.rmsleloss = RMSLELoss()
        self.l1loss = nn.L1Loss

    def forward(self, x):
        out = self.fully(x) 
        out = self.out_layer(out)

        return out
    





class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()

        input_size, hidden_size, num_layers, output_size = config.input_size,config.lstm_hidden_size,config.lstm_num_layers,config.num_labels

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.mseloss = nn.MSELoss()
        self.huberloss = nn.HuberLoss()
        self.rmsleloss = RMSLELoss()
        self.l1loss = nn.L1Loss
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float32).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float32).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out
