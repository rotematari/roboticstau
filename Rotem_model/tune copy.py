import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import LambdaLR
from ray import tune
from ray.tune import Trainable
import logging
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train import report,Checkpoint
from ray.train.torch import TorchTrainer

# Import your modules
from main import parse_args, set_device
from data.data_processing import DataProcessor
from models.models import CNNLSTMModel, TransformerModel, TemporalConvNet, CNN2DLSTMModel
from utils.utils import train, model_eval_metric, test

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RayTuneTrainable(Trainable):
    def setup(self, config):
        self.config = parse_args()
        self.device = set_device()

        # Setting the hyperparameters from the config
        for key, value in config.items():
            setattr(self.config, key, value)
        
        self.data_processor = DataProcessor(self.config)
        self.data_processor.load_data()
        self.data_processor.preprocess_data()

        self.config.num_labels = self.data_processor.label_size

        self.model = CNNLSTMModel(self.config).to(self.device)
        self.train_loader, self.val_loader, self.test_loader = self.data_processor.get_data_loaders()
        self.criterion = MSELoss()
        self.optimizer, self.scheduler = self.setup_optimizer_scheduler()

    def setup_optimizer_scheduler(self):
        if self.model.name in ["TransformerModel", "TemporalConvNet"]:
            warmup_steps = 4000
            initial_lr = 1e-4
            lr_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
            optimizer = Adam(self.model.parameters(), lr=initial_lr, weight_decay=self.config.weight_decay)
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            scheduler = None
        return optimizer, scheduler

    def step(self):

        train_loss = self.run_training_epoch()
        val_loss = self.run_validation_epoch()
        
        checkpoint_data = {
            
            "net_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        report({"loss": val_loss},checkpoint=checkpoint)
        return {"loss": val_loss}

    def run_training_epoch(self):
        train_loss = 0
        self.model.train()
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.scheduler and self.model.name in ["TransformerModel", "TemporalConvNet"]:
                self.scheduler.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def run_validation_epoch(self):
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def compute_loss(self, outputs, targets):
        if self.config.sequence:
            if outputs.dim() == 3:
                return self.criterion(outputs[:, -1:, :].squeeze(), targets[:, -1:, :].squeeze())
            else:
                return self.criterion(outputs, targets[:, -1, :])
        else:
            return self.criterion(outputs, targets)

def train_model(config):
    trainer = RayTuneTrainable(config)

    for i in range(config['num_epochs']):  # 10 epochs
        trainer.step()

if __name__ == '__main__':
    
    # Configuration space for hyperparameters
    search_space = {
        'sequence_length': tune.randint(1,100),
        'dropout': tune.uniform(0.2, 0.5),
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'weight_decay': tune.loguniform(1e-6, 1e-3),
        'num_epochs': tune.randint(10,100),
        'batch_size': tune.choice([8, 16, 32,64]),
        'window_size': tune.randint(20 ,50),
        'lstm_hidden_size' : tune.randint(64 ,512),
        'lstm_num_layers' : tune.randint(2 ,10),
    }
    # test     
    # search_space = {
    #     'sequence_length': 2,
    #     'dropout': 0.2,
    #     'learning_rate': 1e-3,
    #     'weight_decay': 1e-3,
    #     'num_epochs': 100,
    #     'batch_size': 32,
    #     'window_size': 20,
    #     'lstm_hidden_size' : 128,
    #     'lstm_num_layers' : 1 ,
    # }


    # trainable(search_space=search_space)
    num_samples = 1

    # Define the scheduler and search algorithm
    scheduler = ASHAScheduler(
        max_t=100,  # max number of epochs
        grace_period=1,
        reduction_factor=2)




    # Run the experiment
    tuner = tune.Tuner(trainable=train_model,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    # results = tune.run(
    #     train_model,
    #     resources_per_trial={"cpu": 20, "gpu": 1 if torch.cuda.is_available() else 0},
    #     metric="loss",
    #     mode="min",
    #     config=search_space,
    #     num_samples=num_samples,  # number of different hyperparameter combinations to try
    #     scheduler=scheduler
    # )

    # Get the hyperparameters of the best trial run
    best_trial = results.get_best_result("loss", "min")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics}")


