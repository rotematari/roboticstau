import torch.nn as nn


class fully_conected_classification(nn.Module):
    def __init__(self, input_size, args_config):
        super(fully_conected_classification, self).__init__()
        self.input_size = input_size

        self.dropout1 = nn.Dropout1d(args_config.dropout_1)
        self.dropout2 = nn.Dropout1d(args_config.dropout_2)
        self.dropout3 = nn.Dropout1d(args_config.dropout_3)

        self.batch_norm_1 = nn.BatchNorm1d(args_config.hidden_size_1)
        self.batch_norm_2 = nn.BatchNorm1d(args_config.hidden_size_2)
        self.batch_norm_3 = nn.BatchNorm1d(args_config.hidden_size_3)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.l1 = nn.Linear(input_size, args_config.hidden_size_1)
        self.l2 = nn.Linear(args_config.hidden_size_1, args_config.hidden_size_2)
        self.l3 = nn.Linear(args_config.hidden_size_2, args_config.hidden_size_3)
        self.l4 = nn.Linear(args_config.hidden_size_3, args_config.num_classes)

    def forward(self, x):
        # layer 1
        out = self.relu1(self.l1(x))
        out = self.batch_norm_1(out)
        out = self.dropout1(out)
        # layer 2
        out = self.relu2(self.l2(out))
        out = self.batch_norm_2(out)
        out = self.dropout2(out)
        # layer 3
        out = self.relu3(self.l3(out))
        out = self.batch_norm_3(out)
        out = self.dropout3(out)
        # outer layer
        out = self.l4(out)
        return out
