import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ray import tune
from ray.train import report
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Ray Tune Trainable
class TrainMNIST(Trainable):
    def setup(self, config):
        self.device = "cpu"
        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config["lr"])
        self.criterion = nn.CrossEntropyLoss()

        # Load MNIST data
        self.train_loader = DataLoader(
            datasets.MNIST(
                './mnisttestdata', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=int(config["batch_size"]), shuffle=True
        )

    def step(self):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return {"loss": total_loss / len(self.train_loader)}

def train_mnist(config):
    trainer = TrainMNIST(config)
    for i in range(10):  # 10 epochs
        trainer.step()

if __name__ == "__main__":
    config = {
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "batch_size": tune.choice([32, 64, 128])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run(
        train_mnist,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler
    )

    print("Best config: ", analysis.best_config)
