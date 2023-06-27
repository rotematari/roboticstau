import torch
import numpy as np
import wandb

# def log(loss, accuracy, epoch):

def train_epoch(epoch, model, train_loader, optimizer, args_config, criterion, device):
    model.cuda()
    wandb.watch(model, criterion, log="all")
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
        wandb.log({"epoch": epoch, "loss": loss.item()})

    print(f' Loss: {loss.item():.4f}')

    return model, optimizer


def check_accuracy( model, validation_loader, optimizer, args_config, criterion, device):
    model.cuda()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]
        class_acc = [0 for i in range(4)]
        count = 0
        for batch_index, (input, labels) in enumerate(validation_loader):
            labels = labels.to(device)
            input = input.to(device)
            outputs = model(input)

            _, predicted = torch.max(outputs.data, 1)

            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum()
            acc = n_correct / n_samples
            wandb.log({"acc":acc})
                # print(f'accurcy :{acc} %')
            count += 1
            # for i in range(args_config.batch_size):
            #     pred = predicted[i]
            #     if (labels[i] == pred):
            #         n_correct += 1

            for j in range(args_config.batch_size):
                label = labels[j]
                pred = predicted[j]
                if (label == pred):
                    n_class_correct[int(label)] += 1
                n_class_samples[int(label)] += 1

    # try:
    for k in range(args_config.num_classes):
        if not n_class_samples[k] == 0:
            class_acc[k] = 100.0 * n_class_correct[k] / n_class_samples[k]
            print(f'Accuracy of {k}: {class_acc[k]} %')
    # except:
    #     print("no match")

    print(f'accurcy :{acc} %')
    return acc
