import torch.nn as nn
import torch


class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()

        input_size, hidden_size, num_layers, output_size , dropout = config.input_size,config.lstm_hidden_size,config.lstm_num_layers,config.num_labels,config.dropout

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        
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


class CNN_LSTMModel(nn.Module):
    def __init__(self, config):
        super(CNN_LSTMModel, self).__init__()

        input_size, hidden_size, num_layers, output_size , dropout = config.input_size,config.lstm_hidden_size,config.lstm_num_layers,config.num_labels,config.dropout

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # cnn layer
        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=1)
        
        # TODO: add an option for more layers of CNN
        # TODO: add max poll

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        
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
        
        # CNN layer
        x = self.cnn(x.unsqueeze(1))


        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

class CNN2D_LSTMModel(nn.Module):
    def __init__(self, config):
        super(CNN2D_LSTMModel, self).__init__()

        input_size, hidden_size, num_layers, output_size, dropout, num_features = config.input_size, config.lstm_hidden_size, config.lstm_num_layers, config.num_labels, config.dropout, config.num_features

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN2D layer
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=(config.sequence_length, 1))

        # LSTM layer
        self.lstm = nn.LSTM(num_features * hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.mseloss = nn.MSELoss()
        self.huberloss = nn.HuberLoss()
        self.l1loss = nn.L1Loss()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)

        # CNN2D layer
        x = self.cnn(x)
        
        # Reshape for LSTM layer
        x = x.view(x.size(0), -1, self.hidden_size).permute(0, 2, 1)
        
        # LSTM layer
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # Fully connected layer
        out = self.fc(out[:, -1, :])
        return out