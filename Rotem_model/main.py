import torch
import argparse
import yaml
import wandb
import os 

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data.data_processing import DataProcessor
from models.models import CNN_LSTMModel , TransformerModel, TemporalConvNet
from utils.utils import train, model_eval_metric, test


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Running on the CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--label_norm', type=bool, help='if to normalize labels')
    parser.add_argument('--n_layer', type=int, help='The number of hidden layers.')
    parser.add_argument('--lstm_hidden_size', type=int, help='The size of each hidden layer.')
    parser.add_argument('--lstm_num_layers', type=int, help='The number of layers.')
    parser.add_argument('--dropout', type=float, help='The dropout rate for each hidden layer.')
    parser.add_argument('--learning_rate', type=float, help='The learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, help='The number of epochs to train for.')
    parser.add_argument('--weight_decay', type=float, help='The weight decay for the optimizer.')
    parser.add_argument('--batch_size', type=int, help='The size of batches.')
    parser.add_argument('--window_size', type=int, help='The size of batches.')
    parser.add_argument('--sequence_length', type=int, help='The sequence length.')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return argparse.Namespace(**config)


def main():
    device = set_device()

    config = parse_args()
    run = None

    if config.wandb_on:
        run = wandb.init(project="FMG_LSTM", config=config)
        config = wandb.config
        wandb.define_metric("Val_loss", summary="min")
        wandb.define_metric("Train_loss", summary="min")

    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.preprocess_data()

    config.num_labels = data_processor.label_size

    if config.plot_data:
        data_processor.plot(from_indx=0, to_indx=1000)

    model = TemporalConvNet(config).to(device)
    print(model)

    train_loader, val_loader, test_loader = data_processor.get_data_loaders()

    if config.pre_trained:
        model.load_state_dict(torch.load(config.best_model)['model_state_dict'])
    else:
        # why did you return train and val if you are not using it?
        train_losses, val_losses = train(config=config, train_loader=train_loader,
                                         val_loader=val_loader, model=model, device=device, wandb_run=run)

    # Test
    eval_metric = model_eval_metric(config, model, test_loader,
                                    data_processor.label_max_val, data_processor.label_min_val,
                                    device=device, wandb_run=run)

    if config.wandb_on:
        wandb.log({"eval_metric": eval_metric})
        wandb.finish()

    print(f'eval_metric {eval_metric}')


if __name__ == '__main__':
    main()
