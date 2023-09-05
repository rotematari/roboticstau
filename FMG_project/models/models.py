import torch.nn as nn
import torch


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
