# configs/env_config.py

import torch
import os
from pathlib import Path 

IS_COLAB = os.getenv('COLAB_ENV') == '1'
print(f"Running in {'Colab' if IS_COLAB else 'Local'} environment.")

PROJECT_ROOT = Path(__file__).parent.parent 
BASE_DIR = Path("/content") if IS_COLAB else PROJECT_ROOT
DRIVE_DIR = Path("/content/drive/MyDrive") if IS_COLAB else PROJECT_ROOT

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'dataset_path': BASE_DIR / 'datasets' / 's3dis' / 's3dis', 
    'num_points': 4096,
    'num_classes': 13,
    'feature_dim': 10,
    
    'learning_rate': 1e-3, 
    'batch_size': 4,       
    'num_epochs': 120,
    'finetune_epoch': 10, 
    'finetune_lr': 1e-5,  
    'weight_decay': 1e-4, 
    'accumulation_steps': 4,
    
    'main_input_dim': 32,
    'knn_param': 16,   
    'interpolation_k': 3,
    'dropout_param': 0.1,
    'num_heads': 4, 

    'neighbor_finder': 'knn', 
    'search_radius': 0.1,     

    'kpconv_radius': 0.1,      
    'kpconv_kernel_size': 15,  
    'kpconv_output_dim': 64,   
    
    'checkpoint_dir': DRIVE_DIR / "saved_models" , 
    
    'stages_config': [
        {'hidden_dim': 32, 'num_layers': 1, 'downsample_ratio': None}, 
        {'hidden_dim': 64, 'num_layers': 2, 'downsample_ratio': 0.25}, 
        {'hidden_dim': 128, 'num_layers': 4, 'downsample_ratio': 0.25}, 
        {'hidden_dim': 256, 'num_layers': 2, 'downsample_ratio': 0.25}, 
        {'hidden_dim': 512, 'num_layers': 2, 'downsample_ratio': 0.25}, 
    ]
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)