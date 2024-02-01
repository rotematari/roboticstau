import torch
import argparse
import yaml
import wandb
import os 

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data.data_processing import DataProcessor
from models.models import CNNLSTMModel , TransformerModel, TemporalConvNet, CNN2DLSTMModel
from utils.utils import train, model_eval_metric, test_model, euclidian_end_effector_eror , plot_results


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
    parser.add_argument('--model', type=str, help='network to use')
    parser.add_argument('--norm', type=str, help='normalization technic')
    parser.add_argument('--n_head', type=str, help='n_attentiopn head')


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
        wandb_run = wandb.init(project="FMG_LSTM", config=config)
        
        config = wandb.config
        wandb.define_metric("Val_loss", summary="min")
        wandb.define_metric("Train_loss", summary="min")
        wandb.define_metric("Val_Avarege_Euclidian_End_Effector_Eror", summary="min")
        wandb.define_metric("Val_Max_Euclidian_End_Effector_Eror", summary="min")

    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.preprocess_data()

    config.num_labels = data_processor.label_size

    if config.plot_data:
        data_processor.plot(from_indx=0, to_indx=10000)

    if config.model == 'CNN_LSTMModel':
        model = CNNLSTMModel(config).to(device)
    elif config.model == 'TransformerModel':
        model = TransformerModel(config).to(device)
    
    print(model)

    train_loader, val_loader, test_loader = data_processor.get_data_loaders()

    if config.pre_trained:
        model.load_state_dict(torch.load(config.best_model)['model_state_dict'])
        # Test
        eval_metric = model_eval_metric(config, model, test_loader,data_processor=data_processor,
                                    device=device, wandb_run=run)
    else:
        
        best_model_checkpoint_path ,best_model_checkpoint= train(config=config, train_loader=train_loader,
                    val_loader=val_loader, 
                    model=model,
                    data_processor=data_processor, 
                    device=device, 
                    wandb_run=run)
        

        model.load_state_dict(torch.load(best_model_checkpoint_path)['model_state_dict'])

        # Test

        test_loss,avg_iter_time, test_avg_location_eror,test_avg_euc_end_effector_eror,test_max_euc_end_effector_eror = test_model(
            model=model,
            config=config,
            data_loader=test_loader,
            data_processor=data_processor,
            device=device,
        )
        

        plot_results(config=config,data_loader=test_loader,model=model,data_processor=data_processor,device=device)

            # print("eror in plot")
    print(f'Test_Loss: {test_loss} \n Test_Avarege_Location_Eror:{test_avg_location_eror} \n Test_Max_Euclidian_End_Effector_Eror : {test_max_euc_end_effector_eror} \n Test_Avarege_Euclidian_End_Effector_Eror: {test_avg_euc_end_effector_eror}')
    if config.wandb_on:
        
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(best_model_checkpoint_path)
        wandb.log_artifact(artifact)
        wandb.log({"Test_Loss": test_loss,"Test_Avarege_Location_Eror":test_avg_location_eror ,"Test_Max_Euclidian_End_Effector_Eror" : test_max_euc_end_effector_eror , "Test_Avarege_Euclidian_End_Effector_Eror": test_avg_euc_end_effector_eror})
        wandb.finish()

    


if __name__ == '__main__':
    main()
