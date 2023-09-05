import os
import torch 
import argparse
import yaml

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from data.data_processing import DataProcessor
from models.models import LSTMModel
from utils.utils import train, model_eval_metric,test

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--config', type=str,default=r'config.yaml', help='Path to the configuration file.')
    parser.add_argument('--label_norm', type=bool, help='if to normelize labels ')
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

    # Data Processing
    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.preprocess_data()
    
    if config.plot_data:
        data_processor.plot(from_indx=800,to_indx=1200)
    
    # Model Initialization
    model = LSTMModel(config)

    model= model.to(device=device)
    print(model)

    # Get DataLoaders
    train_loader, val_loader,test_loader = data_processor.get_data_loaders()

    # Training and Validation
    train_losses, val_losses = train(config=config,train_loader=train_loader
                                    ,val_loader=val_loader,model=model,device=device )


    #test
    eval = model_eval_metric(config,model,test_loader,
                             data_processor.label_max_val,data_processor.label_min_val,
                             device=device)

    test_loss = test(model=model,config=config,
                     test_loader=test_loader,
                     device=device)
    

    print(f'eval_metric {eval}')