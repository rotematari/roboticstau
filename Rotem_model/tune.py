
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import  session
from ray.train import Checkpoint,get_checkpoint
import torch 
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
# from main import parse_args,set_device
from data.data_processing import DataProcessor
from models.models import CNNLSTMModel , TransformerModel, TemporalConvNet, CNN2DLSTMModel
from utils.utils import train, model_eval_metric, test
import time
import argparse
import yaml
from types import SimpleNamespace
import tempfile
import os 
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Running on the CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device



def trainable(config):
    # session.report(
    #         {"loss": 1})

    # Convert dictionary to an object
    config = SimpleNamespace(**config)

    # session.report(
    #         {"loss": config})
    device = set_device()

    # # Setting the hyperparameters from the search space
    # for key, value in search_space.items():
    #     setattr(config, key, value)
    # config.num_epochs = search_space['num_epochs']

    # print(config)

    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.preprocess_data()

    config.num_labels = data_processor.label_size


    model = CNNLSTMModel(config).to(device)


    train_loader, val_loader, test_loader = data_processor.get_data_loaders()
    criterion = MSELoss()
    if model.name == "TransformerModel" or model.name == "TemporalConvNet":
        # Learning rate warm-up
        warmup_steps = 4000
        initial_lr = 1e-4  # Starting learning rate

        # Custom lambda for learning rate schedule
        lr_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)

        optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=config.weight_decay)

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_losses = []
    val_losses = []
    best_val_loss = 0


    

    if get_checkpoint():
        loaded_checkpoint = get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    # print(f"training starts from epoch {start_epoch}")

    # Train the network
    
    for epoch in range(config.num_epochs):

        # Initialize the epoch loss and accuracy
        train_loss = 0
        model.train()
        # Train on the training set
        for inputs, targets in train_loader:

            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            if config.sequence:
                if outputs.dim() == 3:
                    loss = criterion(outputs[:,-1:,:].squeeze(), targets[:,-1:,:].squeeze())
                else:
                    loss = criterion(outputs, targets[:,-1,:])
            else:
                loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            if model.name == "TransformerModel" or model.name == "TemporalConvNet":
                scheduler.step()  # Update the learning rate

            # Update the epoch loss and accuracy
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"train loss{train_loss}")
        train_losses.append(train_loss)
        val_loss = 0
        total_time = 0
        # Evaluate on the validation set
        with torch.no_grad():
            model.eval()
            
            for i,(inputs, targets) in enumerate(val_loader):
                
                # start_time = time.time()
                inputs = inputs.to(device=device)
                targets = targets.to(device=device)

                outputs = model(inputs)
                if config.sequence:
                    if outputs.dim() == 3:
                        v_loss = criterion(outputs[:,-1:,:].squeeze(), targets[:,-1:,:].squeeze())
                    else:
                        v_loss = criterion(outputs, targets[:,-1,:])
                else:
                    v_loss = criterion(outputs, targets)

                val_loss += v_loss.item()
                # end_time = time.time()
                # total_time += (end_time - start_time)

            
            val_loss /= len(val_loader)
            # avg_iter_time = total_time/len(val_loader)
        print ( f"val loss {val_loss}")
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        # checkpoint.update(checkpoint_data)
        # checkpoint = Checkpoint.get_metadata(checkpoint_data)
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        session.report(
            {"loss": val_loss,},
            checkpoint=checkpoint,
        )

        epoch += 1



if __name__ == '__main__':
    training_config = {
    "sensor_location": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "fmg_index": ['S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45'],
    "velocity_label_index": ['VCx', 'VCy', 'VCz', 'VSx', 'VSy', 'VSz', 'VEx', 'VEy', 'VEz', 'VWx', 'VWy', 'VWz'],
    "session_time_stamp": ['session_time_stamp'],
    "label_index": ['MCx', 'MCy', 'MCz', 'MSx', 'MSy', 'MSz', 'MEx', 'MEy', 'MEz', 'MWx', 'MWy', 'MWz'],
    
    "input_size": 28,
    "num_labels": 9,
    "sample_speed": 100,
    

    'sequence_length': tune.randint(1,100),
    'dropout': tune.uniform(0.2, 0.5),
    'learning_rate': tune.loguniform(1e-5, 1e-3),
    'weight_decay': tune.loguniform(1e-6, 1e-3),
    "num_epochs": tune.randint(1,2),
    'batch_size': tune.choice([8, 16, 32,64]),
    'window_size': tune.randint(20 ,50),
    'lstm_hidden_size' : tune.randint(64 ,256),
    'lstm_num_layers' : tune.randint(2 ,10),
    
    "data_path": "/home/robotics20/roboticstau/Rotem_model/data/full_muvment_clean",
    "model_path": "./models/saved_models",
    
    "test_size": 0.01,
    "val_size": 0.3,
    "random_state": 42,
    "shuffle_train": False,
    "wandb_on": False,
    "norm": "minmax",
    "norm_labels": True,
    "plot_data": True,
    "plot_pred": True,
    "pre_trained": False,
    "with_velocity": False,
    "sequence": True
    }

    # Configuration space for hyperparameters
    # search_space = {
    #     'sequence_length': tune.randint(1,100),
    #     'dropout': tune.uniform(0.2, 0.5),
    #     'learning_rate': tune.loguniform(1e-5, 1e-3),
    #     'weight_decay': tune.loguniform(1e-6, 1e-3),
    #     "num_epochs": tune.randint(10,100),
    #     'batch_size': tune.choice([8, 16, 32,64]),
    #     'window_size': tune.randint(20 ,50),
    #     'lstm_hidden_size' : tune.randint(64 ,512),
    #     'lstm_num_layers' : tune.randint(2 ,10),
    # }


    num_samples = 1

    # Define the scheduler and search algorithm
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,  # max number of epochs
        grace_period=1,
        reduction_factor=2)

    # # Start Ray
    ray.init(ignore_reinit_error=True)

    # Run the experiment
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 20, "gpu": 1},
        config=training_config,
        num_samples=num_samples,  # number of different hyperparameter combinations to try
        scheduler=scheduler
    )


    # Get the hyperparameters of the best trial run
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    # # Save the best model checkpoint path
    # best_checkpoint_path = best_trial.last_result['checkpoint']

    # # You can now load the best model using this checkpoint
    # best_model_state = torch.load(best_checkpoint_path)
