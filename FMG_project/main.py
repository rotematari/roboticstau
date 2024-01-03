import os
import torch 
import argparse
import yaml
import wandb

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data.data_processing import DataProcessor
from models.models import LSTMModel,CNN_LSTMModel
from utils.utils import train, model_eval_metric,test

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--config', type=str,default=r'config.yaml', help='Path to the configuration file.')
    parser.add_argument('--label_norm', type=bool, help='if to normelize labels ')
    # Add arguments for the configuration options
    # parser.add_argument('--input_size', type=int, default=27, help='The size of the input layer.')
    # parser.add_argument('--num_labels', type=int, default=18, help='The number of output labels.')
    parser.add_argument('--n_layer', type=int, help='The number of hidden layers.')
    parser.add_argument('--lstm_hidden_size', type=int, help='The size of each hidden layer.')
    parser.add_argument('--lstm_num_layers', type=int, help='The number of layers.')
    parser.add_argument('--dropout', type=float, help='The dropout rate for each hidden layer.')

    # parser.add_argument('--multiplier', nargs='+', type=float, help='The multiplier gor hidden state.')
    # Add arguments for the hyperparameters
    parser.add_argument('--learning_rate', type=float, help='The learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, help='The number of epochs to train for.')
    parser.add_argument('--weight_decay', type=float, help='The weight decay for the optimizer.')
    parser.add_argument('--batch_size', type=int, help='The size of batchs.')
    parser.add_argument('--window_size', type=int, help='The size of batchs.')
    parser.add_argument('--sequence_length', type=int, help='The sequence_length.')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # Override yaml configs with command-line arguments if needed
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return argparse.Namespace(**config)




if __name__ == '__main__':
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set the device to the first available CUDA device
        device = torch.device("cuda:0")
        print(f"Running on the CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")


    config = parse_args()
    run = None

    if config.wandb_on:
        run = wandb.init(project="FMG_LSTM",config=config)
        config = wandb.config
        # define a metric we are interested in the minimum of
        wandb.define_metric("Val_loss", summary="min")
        wandb.define_metric("Train_loss", summary="min")

    # Data Processing
    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.preprocess_data()

    config.num_labels = data_processor.label_size
    
    if config.plot_data:
        data_processor.plot(from_indx=0,to_indx=1000)
    
    # Model Initialization
    model = CNN_LSTMModel(config)

    model= model.to(device=device)

    print(model)

    # Get DataLoaders
    train_loader, val_loader,test_loader = data_processor.get_data_loaders()


    if config.pre_trained:
        model.load_state_dict(torch.load(config.best_model)['model_state_dict'])
    else:
        # Training and Validation
        train_losses, val_losses = train(config=config,train_loader=train_loader
                                        ,val_loader=val_loader,model=model,device=device,wandb_run=run)


    #test
    eval = model_eval_metric(config,model,test_loader,
                             data_processor.label_max_val,data_processor.label_min_val,
                             device=device,
                             wandb_run=run)
    if config.wandb_on:
        wandb.log({"eval_metric": eval})
    # test_loss = test(model=model,config=config,
    #                  test_loader=test_loader,
    #                  device=device)
    

    print(f'eval_metric {eval}')
    if config.wandb_on:
        wandb.finish()