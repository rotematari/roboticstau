from os.path import join
from time import strftime, gmtime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import data_loader
from functools import partial
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# import real_time_data

dirpath = '/home/roblab15/Documents/FMG_project/data'
model_dir_path = r'/home/roblab15/Documents/FMG_project/models'
# Hyper-parameters
input_size = 6
num_classes = 4
# num_epochs = 15
items = ['B1', 'B2', 'S1', 'S2', 'S3', 'S4']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        # out = self.relu(out)
        # out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


# ray tune example
def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = NeuralNet(input_size, config["hidden_size"], num_classes)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # pointing dataset
    trainset = data_loader.Data(train=True, dirpath=data_dir, items=items)

    # split train
    test_abs = int(len(trainset) * 0.)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])
    # Data loader
    trainloader = torch.utils.data.DataLoader(dataset=train_subset,
                                              batch_size=int(config["batch_size"]),
                                              shuffle=True)

    valloader = torch.utils.data.DataLoader(dataset=val_subset,
                                            batch_size=int(config["batch_size"]),
                                            shuffle=True)

    for epoch in range(int(config["epoch"])):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def test_accuracy(net, device="cpu", best_batch_size=10):
    testset = data_loader.Data(train=False, dirpath=dirpath, items=items)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=best_batch_size, shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            X, labels = data
            X, labels = X.to(device), labels.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    config = {
        "weight_decay": tune.loguniform(1e-6, 1e-1),
        "epoch": tune.loguniform(10, 200),
        "hidden_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([10, 20, 30, 40, 50, 60]) \
        }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["hidden_size", "learning_rate", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=dirpath),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best batch config: {}", best_trial.config["batch_size"])
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = NeuralNet(input_size, best_trial.config["hidden_size"], num_classes)
    # print(best_trial.checkpoint)
    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device, best_trial.config["batch_size"])
    print("Best trial test set accuracy: {}".format(test_acc))
    save = 0
    save = input("to save model press 1 \n")
    if save == '1':
        model_name = 'model_' + strftime("%d%b%Y%_H:%M", gmtime()) + '.pt'
        torch.save(best_trained_model.state_dict(), join(model_dir_path, model_name))

    # print("real time test \n")
    # run = input("press 1 to run ")
    # while run == '1':
    #     pred = real_time(best_trained_model)
    #     print(pred)


def real_time(best_traind_model):
    X = real_time_data.Data()
    X.x = X.x.to(device)
    outputs = best_traind_model(X.x)
    score, predicted = torch.min(outputs.data, 1)
    # _, predicted = torch.min(score, 1)
    # print(score)
    return predicted


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=1, max_num_epochs=200, gpus_per_trial=1)
