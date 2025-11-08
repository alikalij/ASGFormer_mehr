# main.py

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models.model import ASGFormer
from data.dataset import H5Dataset, PointCloudProcessor, read_file_list, compute_class_weights
from trainer import Trainer 
from configs.env_config import CONFIG 
import os
import h5py

def main():
    hyperparams = CONFIG
    device = torch.device(hyperparams['device'])
    print(f"Using device: {device}")

    dataset_path = hyperparams['dataset_path']
    print(f"Loading dataset from: {dataset_path}")
    train_files = read_file_list(os.path.join(dataset_path, "list", "train5.txt"))
    val_files = read_file_list(os.path.join(dataset_path, "list", "val5.txt"))

    train_processor = PointCloudProcessor(num_points=hyperparams['num_points'], is_training=True)
    val_processor = PointCloudProcessor(num_points=hyperparams['num_points'], is_training=False)
    
    train_dataset = H5Dataset(train_files, train_processor, dataset_path)
    val_dataset = H5Dataset(val_files, val_processor, dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    print("Creating model...")
    model = ASGFormer(
        feature_dim=hyperparams['feature_dim'],
        main_input_dim=hyperparams['main_input_dim'],
        main_output_dim=hyperparams['num_classes'],
        stages_config=hyperparams['stages_config'], 
        knn_param=hyperparams['knn_param'],
        num_heads=hyperparams['num_heads'],
        neighbor_finder=hyperparams['neighbor_finder'], 
        search_radius=hyperparams['search_radius'],     
        interpolation_k=hyperparams['interpolation_k'],
        dropout_param=hyperparams['dropout_param'],
        kpconv_radius=hyperparams['kpconv_radius'],
        kpconv_kernel_size=hyperparams['kpconv_kernel_size']
    )
    print("Setting up criterion, optimizer, and scheduler...")
    try:
        all_train_labels = [h5py.File(os.path.join(dataset_path, f), 'r')['label'][:] for f in train_files]
        class_weights = compute_class_weights(all_train_labels, num_classes=hyperparams['num_classes']).to(device)
        print("Class weights computed.")
    except Exception as e:
        print(f"Warning: Could not compute class weights: {e}. Using uniform weights.")
        class_weights = None 

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams['num_epochs'], eta_min=1e-6) # eta_min کمی کوچکتر

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        config=hyperparams
    )
    
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()