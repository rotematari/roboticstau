import os
from functools import partial

import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

import torch
import torch.utils.data
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

import data_loader
import paramaters
from models import fully_connected as net

class arg_config:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.dropout_1 = config["dropout_1"]
        self.dropout_2 = config["dropout_1"]
        self.dropout_3 = config["dropout_1"]
        self.hidden_size_1 = config["dropout_1"]
        self.hidden_size_2 = config["dropout_1"]
        self.hidden_size_3 = config["dropout_1"]
        self.num_classes = config["dropout_1"]


def main(max_num_epochs=15, gpus_per_trial=1, num_samples=1):
    train_data = data_loader.Data(train=True, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)

    config = {
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "epoch": tune.loguniform(10, 50),
        "hidden_size_1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "hidden_size_2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "hidden_size_3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 8)),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([20, 30, 40, 50, 60]),
        "dropout_1": tune.loguniform(0.01, 0.4),
        "dropout_2": tune.loguniform(0.01, 0.4),
        "dropout_3": tune.loguniform(0.01, 0.4), \
        }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["hidden_size_1", "hidden_size_2", "hidden_size_2", "learning_rate", "batch_size",
                           "weight_decay", "epoch"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_args_config = arg_config(best_trial.config)
    best_trained_model = net(train_data.n_featurs, best_args_config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def train(net, config, device="cpu"):
    args_config = arg_config(config)

    # data sets
    train_data = data_loader.Data(train=True, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)

    # split train
    test_abs = int(len(train_data) * 0.9)
    train_subset, validation_subset = random_split(train_data, [test_abs, len(train_data) - test_abs])

    # torch data loader seperate to batches and shuffel

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=args_config.batch_size, shuffle=True,
                                               drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_subset, batch_size=args_config.batch_size,
                                                    shuffle=True, drop_last=True)

    net = net(train_data.n_featurs, args_config)
    net = net.to(device)
    net.train()
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    for epoch in range(int(config["epoch"])):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):

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
            if i % len(train_loader) / 2 == 0:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(validation_loader, 0):
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
    testset = data_loader.Data(train=False, dirpath=paramaters.parameters.dirpath, items=paramaters.parameters.items)
    testloader = torch.utils.data.DataLoader(testset, batch_size=best_batch_size, shuffle=True, drop_last=True)
    classes = [1, 2, 3, 4]
    n_correct = 0
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

            for i in range(best_batch_size):
                pred = predicted[i]
                if (labels[i] == pred):
                    n_correct += 1

            # correct += (predicted == labels).sum().item()
        acc_total = 100.0 * n_correct / total
        print(f'Accuracy of the network: {acc_total} %')
        for i in range(best_batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[int(label)] += 1
            n_class_samples[int(label)] += 1

    for i in range(4):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

    return acc_total


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(max_num_epochs=15, gpus_per_trial=1, num_samples=1)
