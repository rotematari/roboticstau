from os.path import join
# import real_time_data
from time import gmtime, strftime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import data_loader
import paramaters

dirpath = paramaters.parameters.dirpath
model_dir_path = r'/home/roblab15/Documents/FMG_project/models'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
items = paramaters.parameters.items
# Hyper-parameters
# input_size = 6
hidden_size_1 = 128
hidden_size_2 = 16
hidden_size_3 = 8
num_classes = 4
num_epochs = 5

batch_size = 30
learning_rate = 0.008
weight_decay = 0.0001
dropout = 0.1

classes = ['0', '1', '2', '3']
# pointing dataset
point_data_train = data_loader.Data(train=True, dirpath=dirpath, items=items)
point_data_test = data_loader.Data(train=False, dirpath=dirpath, items=items)

input_size = point_data_train.n_featurs
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=point_data_train,
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=point_data_test,
                                          batch_size=batch_size,
                                          shuffle=True, drop_last=True)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_classes, dropout):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.dropout1 = nn.Dropout1d(0.1)
        self.dropout2 = nn.Dropout1d(0.3)
        self.dropout3 = nn.Dropout1d(0.2)

        self.batchnorm_1 = nn.BatchNorm1d(hidden_size_1)
        self.batchnorm_2 = nn.BatchNorm1d(hidden_size_2)
        self.batchnorm_3 = nn.BatchNorm1d(hidden_size_3)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.l1 = nn.Linear(input_size, hidden_size_1)
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.l3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.l4 = nn.Linear(hidden_size_3, num_classes)

        # self.conv1 = nn.Conv1d()

        # dropout, batchnorm

    def forward(self, x):
        # out = self.dropout(x)
        out = self.l1(x)
        out = self.relu1(out)
        out = self.batchnorm_1(out)
        out = self.dropout1(out)

        out = self.l2(out)
        out = self.relu2(out)
        out = self.batchnorm_2(out)
        out = self.dropout2(out)

        out = self.l3(out)
        out = self.relu3(out)
        out = self.batchnorm_3(out)
        out = self.dropout3(out)

        out = self.l4(out)
        return out


def train_model(train_loader):
    # Loss and optimizer

    # Train the model
    n_total_steps = len(train_loader)
    print(n_total_steps)
    for epoch in range(num_epochs):
        for i, (X, labels) in enumerate(train_loader):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            # images = images.reshape(-1, 28 * 28).to(device)
            # labels = labels.to(device)

            X = X.to(device)
            labels = labels.to(device)
            # print(X.shape, labels.shape)
            # Forward pass
            outputs = model(X)
            labels = labels.long()
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % n_total_steps == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
                # plt.plot(loss.item())
            # if loss <= 0.1:
            #     print("loss small ")
            #     break
        # if loss <= 0.1:
        #     print("loss small ")
        #     break
    # plt.show()


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test_model(test_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]
        for X, labels in test_loader:
            labels = labels.to(device)

            X = X.to(device)
            outputs = model(X)
            # print(outputs.data)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            for i in range(batch_size):
                pred = predicted[i]
                if (labels[i] == pred):
                    n_correct += 1

            # n_correct += (predicted == labels).sum().item()

        # acc = 100.0 * n_correct / n_samples
        # print(f'Accuracy of the network : {acc} %')

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[int(label)] += 1
            n_class_samples[int(label)] += 1


    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(4):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
    return acc


# def predict(NeuralNet, inputs):
#     logits = NeuralNet.forward(inputs)
#     return logits.numpy().argmax(axis=1)


# def real_time():
#     X = real_time_data.Data()
#
#     # X = X.to(device)
#     outputs = model(X.x)
#     # print(outputs.data)
#     # max returns (value ,index)
#     _, predicted = torch.max(outputs.data, 1)
#
#     return predicted

# def plot_cost(loss, num_epouch):

model = NeuralNet(input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_classes, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_model(train_loader)
test_model(test_loader)

save = 0
save = input("to save model press 1 \n")
if save == '1':
    model_name = 'model_' + strftime("%d%b%Y%_H:%M", gmtime()) + '.pt'
    torch.save(model.state_dict(), join(model_dir_path, model_name))
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
# real_t = 1
# while real_t == 1:
#     pred = real_time()
#     print(pred)
