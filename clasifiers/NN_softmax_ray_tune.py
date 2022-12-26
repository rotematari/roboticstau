import os
from functools import partial
from os.path import join
from time import strftime, gmtime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import Dataset
from torch.utils.data import random_split

import data_loader
import paramaters

# import real_time_data

dirpath = paramaters.parameters.dirpath
model_dir_path = r'/home/roblab15/Documents/FMG_project/models'
# Hyper-parameters
input_size = 3
num_classes = 4
# num_epochs = 15
items = paramaters.parameters.items
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['0', '1', '2', '3']

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes, dropout1, dropout2):
        super(NeuralNet, self).__init__()
        self.my_network = torch.nn.Sequential(
            # first
            torch.nn.Linear(input_size, hidden_size_1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size_1),
            torch.nn.Dropout1d(dropout1),

            # second
            torch.nn.Linear(hidden_size_1, hidden_size_2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size_2),
            torch.nn.Dropout1d(dropout2),

            # output layer
            torch.nn.Linear(hidden_size_2, num_classes)
        )
        # self.input_size = input_size
        # self.dropout1 = nn.Dropout1d(dropout)
        # self.dropout2 = nn.Dropout1d(dropout)
        # self.batchnorm = nn.BatchNorm1d(num_features=hidden_size)
        # self.l1 = nn.Linear(input_size, hidden_size)
        # self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()
        # self.relu3 = nn.ReLU()
        # self.l2 = nn.Linear(hidden_size, hidden_size)
        # self.l3 = nn.Linear(hidden_size, num_classes)

        # dropout, batchnorm

    def forward(self, x):
        logits = self.my_network(x)
        # # out = self.dropout1(x)
        # out = self.l1(x)
        # out = self.batchnorm(out)
        # out = self.relu1(out)
        # out = self.dropout2(out)
        # # out = self.l2(out)
        # # out = self.batchnorm(out)
        # # out = self.relu2(out)
        # # out = self.relu3(out)
        # # out = self.dropout(out)
        # # out = self.relu(out)
        # # out = self.relu(out)
        # out = self.l3(out)
        return logits


# train


def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = NeuralNet(input_size, config["hidden_size_1"], config["hidden_size_2"], num_classes, config["dropout1"], config["dropout2"])
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], )

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # pointing dataset
    trainset = data_loader.Data(train=True, dirpath=data_dir, items=items)

    # split train
    test_abs = int(len(trainset) * 0.7)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])
    # Data loader
    trainloader = torch.utils.data.DataLoader(dataset=train_subset,
                                              batch_size=int(config["batch_size"]),
                                              shuffle=True, drop_last=True)

    valloader = torch.utils.data.DataLoader(dataset=val_subset,
                                            batch_size=int(config["batch_size"]),
                                            shuffle=True,
                                            drop_last=True)

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
        testset, batch_size=best_batch_size, shuffle=True, drop_last=True)

    correct = 0
    total = 0
    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]
    with torch.no_grad():
        for data in testloader:
            X, labels = data
            X, labels = X.to(device), labels.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # if predicted == labels:
            #     correct += 1

            correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        print(f'Accuracy of the network: {acc} %')
        for i in range(best_batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[int(label)] += 1
            n_class_samples[int(label)] += 1



    for i in range(4):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

    return acc


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    config = {

        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "epoch": tune.loguniform(10, 50),
        "hidden_size_1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "hidden_size_2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([20, 30, 40, 50, 60]),
        "dropout1": tune.loguniform(0.01, 0.4),
        "dropout2": tune.loguniform(0.01, 0.4), \
        }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["hidden_size_1", "hidden_size_2", "learning_rate", "batch_size", "weight_decay", "epoch"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=dirpath),
        resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
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

    best_trained_model = NeuralNet(input_size, best_trial.config["hidden_size_1"], best_trial.config["hidden_size_2"],
                                   num_classes, best_trial.config["dropout1"], best_trial.config["dropout2"])

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
    main(num_samples=5, max_num_epochs=50, gpus_per_trial=1)
