import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import data_loader

dirpath = '/home/roblab15/Documents/FMG_project/data'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 4  # 28x28
hidden_size = 40
num_classes = 4
num_epochs = 15
batch_size = 30
learning_rate = 0.0001

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='./data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='./data',
#                                           train=False,
#                                           transform=transforms.ToTensor())
# pointing dataset
point_data_train = data_loader.Data(train=True, dirpath=dirpath)
point_data_test = data_loader.Data(train=False, dirpath=dirpath)

# point_data.X_train = torch.from_numpy(point_data.X_train)
# point_data.X_test = torch.from_numpy(point_data.X_test)
# point_data.Y_train = torch.from_numpy(point_data.Y_train)
# point_data.Y_test = torch.from_numpy(point_data.Y_test)
# print(point_data.X_train.shape)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=point_data_train,
                                           batch_size=batch_size,
                                           shuffle=True)
# train_loader_y = torch.utils.data.DataLoader(dataset=point_data.Y_train,
#                                            batch_size=batch_size,
#                                            shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=point_data_test,
                                          batch_size=batch_size,
                                          shuffle=True)

# examples = iter(train_loader)
# x, labels = next(examples)
# print(x, labels)


# examples = iter(train_loader_x)
# example_data2 = next(examples)
#
# for i, j in [example_data1, example_data2], range(1) :
#     plt.subplot(1, 1, j + 1)
#     plt.plot(i)
# plt.show()


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# train_loader = [train_loader_x, train_loader_y]

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

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for X, labels in test_loader:
        # images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        X = X.to(device)
        outputs = model(X)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
