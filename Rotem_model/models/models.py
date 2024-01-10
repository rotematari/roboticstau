import torch.nn as nn
import torch
from torch.nn.utils import weight_norm



class CNNLSTMModel(nn.Module):
    def __init__(self, config):
        super(CNNLSTMModel, self).__init__()
        self.name = "CNN_LSTMModel"
        self.config = config 
        input_size, hidden_size, num_layers, output_size, dropout,sequence_length = (
            config.input_size,
            config.lstm_hidden_size,
            config.lstm_num_layers,
            config.num_labels,
            config.dropout,
            config.sequence_length
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if not config.sequence:
            sequence_length = 1
        self.conv1 = nn.Conv1d(in_channels=sequence_length, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout1d(dropout)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True,
                            )

        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), output_size)

    def forward(self, x):


        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)

        if not self.config.sequence:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batch_norm(x)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        out = self.fc2(out)
        out = self.fc3(out)

        return out


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.name = "TransformerModel"
        input_size, hidden_size, num_layers, output_size, dropout = (
            config.input_size,
            config.lstm_hidden_size,
            config.lstm_num_layers,
            config.num_labels,
            config.dropout,
        )

        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
            num_layers=num_layers
        )
        
        self.fc1 = nn.Linear(hidden_size,int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        if not self.config.sequence:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)  # Change the sequence length to be the first dimension
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change it back to the original shape
        x = self.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        # print(x)
        return x
    



# not for use at the moment 
#TODO add TCNmodel 
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation,))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self,config, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.name = "TemporalConvNet"
        num_inputs = config.input_size
        dropout = config.dropout
        num_channels = config.num_channels
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.flattn(1)
        # x = x.unsqueeze(0)
        x = x.permute(0,2,1)
        x = self.network(x)
        x = x.permute(0,2,1)
        # x = x.squeeze(0)
        return x
class CNN2DLSTMModel(nn.Module):
    def __init__(self, config):
        super(CNN2DLSTMModel, self).__init__()
        self.name = "CNN2DLSTMModel"
        self.config = config 
        input_size, hidden_size, num_layers, output_size, dropout = (
            config.input_size,
            config.lstm_hidden_size,
            config.lstm_num_layers,
            config.num_labels,
            config.dropout,
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(hidden_size)
        self.drop = nn.Dropout2d(dropout) 



        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True,
                            # peephole=True
                            )

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):

        # if self.config.sequence:
        #     x = x.flatten(1)
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)

        # if self.config.sequence:
        #     x = x.flatten(1)
        # x = x.permute(1,0,2)
        x = self.conv1(x.unsqueeze(1))
        x = self.relu(x)
        x = self.batch_norm(x)
        # x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.batch_norm(x)
        # x = self.pool2(x)
        # x = x.squeeze(1)
        # x = x.permute(1,0,2)

        # x = x.flatten(1)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out