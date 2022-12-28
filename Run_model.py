import torch
import numpy as np
import wandb


def train_epoch(epoch, model, train_loader, optimizer, args_config, logging, criterion, device):
    for batch_index, (input, labels) in enumerate(train_loader):
        input = input.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input)
        labels = labels.long()
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(epoch, model, validation_loader, optimizer, args_config, logging, criterion, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]
        for batch_index, (input, labels) in enumerate(validation_loader):
            labels = labels.to(device)
            input = input.to(device)

            outputs = model(input)
            _, predicted = torch.max(outputs.data, 1)

            n_samples += labels.size(0)
            n_correct += (predicted == outputs).sum()
            acc = n_correct / n_samples * 100
            print(f'accurcy :{acc} %\n')

            for i in range(args_config.batch_size):
                pred = predicted[i]
                if (labels[i] == pred):
                    n_correct += 1

        for i in range(args_config.batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[int(label)] += 1
            n_class_samples[int(label)] += 1
    class_acc = []
    for i in range(4):
        class_acc[i] = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {i}: {class_acc[i]} %')
    return acc
