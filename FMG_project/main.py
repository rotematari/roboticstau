import os

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
from data.data_processing import DataProcessor
from models.models import LSTMModel
from utils.utils import train, validate

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--config', type=str,default=r'config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return argparse.Namespace(**config)




if __name__ == '__main__':

    
    config = parse_args()

    # Data Processing
    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.preprocess_data()

    # Model Initialization
    model = LSTMModel(config)

    # Get DataLoaders
    train_loader, val_loader = data_processor.get_data_loaders()

    # Training and Validation
    train_losses, val_losses = train(config, model, train_loader)
    validate(config, model, val_loader)