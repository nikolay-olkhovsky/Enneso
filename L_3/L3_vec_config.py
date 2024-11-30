import wandb

sweep_config = {
    'method': 'bayes',
    'name' : 'face2_dim5',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'early_terminate':{
        'type': 'hyperband',
        'max_iter': 10,
        's' : 2
    },
    'parameters': {
        'input_activation' : {
            'values': ['relu', 'tanh', 'sigmoid']    
        },
        'input_dropout': {
            'values': [0., 0.15, 0.30]
        },
        'hidden_1_dense': {
            'values': [512, 1024, 2048, 4096, 8192]
        },
        'hidden_1_activation' : {
            'values': ['relu', 'tanh', 'sigmoid']    
        },
        'hidden_1_dropout': {
            'values': [0., 0.15, 0.30, 0.45]
        },
        'middle_number': {
            'values': [0, 1, 2, 3]
        },
        'middle_dense': {
            'values': [512, 1024, 2048, 4096, 8192]
        },
        'middle_activation' : {
            'values': ['relu', 'tanh', 'sigmoid']    
        },
        'middle_dropout': {
            'values': [0., 0.15, 0.30, 0.45]
        },
        'batch_size': {
            'values': [50, 100, 256]
        },
        'learning_rate':{
            'values': [0.0001, 0.000075, 0.00005, 0.000025, 0.00001, 0.0000075]
        }
    }
}
wandb.login()
wandb.sweep(sweep_config, project="BSF-ANN")