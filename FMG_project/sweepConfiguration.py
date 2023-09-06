import wandb 


sweep_configuration ={
    'method': 'random',
    # project this sweep is part of 
    'project':'arm_Modeling' ,
    
    
    'metric': 
    {   'name': 'val_loss',
        'goal': 'minimize',
        # 'target': 0.9 
        
        },
    'parameters': 
    {
        # 'n_layer': {'distribution': 'int_uniform', 'min': 5, 'max': 50},
        # 'hidden_size': {'distribution': 'int_uniform', 'min': [50,50,50], 'max': [300,300,300]}},
        # 'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.7},
        'weight_decay': {'distribution': 'uniform', 'min': 1e-6, 'max': 1e-4},
        'learning_rate': {'distribution': 'uniform', 'min': 1e-5, 'max': 1e-3},
        'batch_size': {'distribution': 'int_uniform', 'min': 2, 'max': 15},
        'multiplier':{'distribution': 'uniform', 'min': 1.5, 'max': 5},
        
    },}