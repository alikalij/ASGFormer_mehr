# main.py
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import os
import h5py

# Import project modules
from models.model import ASGFormer
from data.dataset import H5Dataset, PointCloudProcessor, read_file_list, compute_class_weights
from train import train_model
from configs.env_config import CONFIG # Import centralized config

def main():
    # --- تمام تنظیمات از فایل کانفیگ خوانده می‌شوند ---
    hyperparams = CONFIG
    device = torch.device(hyperparams['device'])
    print(f"Starting main script on device: {device}")

    # --- آماده‌سازی دیتاست ---
    dataset_path = hyperparams['dataset_path']
    try:
        train_files_list_path = os.path.join(dataset_path, "list", "train5.txt")
        val_files_list_path = os.path.join(dataset_path, "list", "val5.txt")
        print(f"Reading train file list from: {train_files_list_path}")
        train_files = read_file_list(train_files_list_path)
        print(f"Reading validation file list from: {val_files_list_path}")
        val_files = read_file_list(val_files_list_path)
        if not train_files or not val_files:
             raise FileNotFoundError("Train or validation file list is empty.")
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset file lists at {dataset_path}/list. Details: {e}")
        print("Please ensure the dataset structure is correct and paths in env_config.py are valid.")
        return # Exit if dataset lists are not found

    processor = PointCloudProcessor(num_points=hyperparams['num_points'])
    
    # Create Datasets
    try:
        train_dataset = H5Dataset(train_files, processor, dataset_path)
        val_dataset = H5Dataset(val_files, processor, dataset_path)
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print("Warning: Train or validation dataset is empty after processing file list.")
    except Exception as e:
        print(f"Error creating H5Dataset instances: {e}")
        return

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=hyperparams['num_workers'],
        pin_memory=hyperparams['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams['batch_size'], # Use same batch size for validation
        shuffle=False,
        num_workers=hyperparams['num_workers'],
        pin_memory=hyperparams['pin_memory']
    )

    # --- ساخت مدل ---
    print("Initializing model...")
    model = ASGFormer(
        feature_dim=hyperparams['feature_dim'],
        main_input_dim=hyperparams['main_input_dim'],
        main_output_dim=hyperparams['num_classes'],
        stages_config=hyperparams['stages_config'],
        knn_param=hyperparams['knn_param'],
        dropout_param=hyperparams['dropout_param']
    ).to(device)
    print("Model initialized successfully.")

    # --- تعریف تابع هزینه، بهینه‌ساز و زمان‌بند ---
    # محاسبه وزن کلاس‌ها (فقط بر اساس داده‌های آموزشی)
    try:
        print("Calculating class weights...")
        # Check if train_files is not empty before proceeding
        if train_files:
            all_train_labels = []
            for f in train_files:
                try:
                    h5_path = os.path.join(dataset_path, f)
                    with h5py.File(h5_path, 'r') as hf:
                        # Ensure 'label' dataset exists
                        if 'label' in hf:
                             all_train_labels.append(hf['label'][:].flatten())
                        else:
                             print(f"Warning: 'label' dataset not found in {h5_path}")
                except Exception as e:
                    print(f"Warning: Could not read labels from {h5_path}. Error: {e}")
            
            if all_train_labels: # Only compute weights if labels were loaded
                class_weights = compute_class_weights(all_train_labels, num_classes=hyperparams['num_classes']).to(device)
                print(f"Class weights computed: {class_weights}")
            else:
                 print("Warning: No labels found to compute class weights. Using uniform weights.")
                 class_weights = torch.ones(hyperparams['num_classes'], device=device)
        else:
             print("Warning: Train file list is empty. Using uniform class weights.")
             class_weights = torch.ones(hyperparams['num_classes'], device=device)

    except Exception as e:
        print(f"Error calculating class weights: {e}. Using uniform weights.")
        class_weights = torch.ones(hyperparams['num_classes'], device=device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=hyperparams['label_smoothing']
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    # Scheduler with warm-up (optional but often helpful)
    # Example: Linear warm-up for 5 epochs then Cosine Annealing
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    #     torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5),
    #     torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams['num_epochs']-5, eta_min=1e-6)
    # ], milestones=[5])
    
    # Simple Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hyperparams['num_epochs'], # T_max is the number of epochs
        eta_min=1e-6 # Minimum learning rate
    )
    print("Criterion, Optimizer, and Scheduler initialized.")

    # --- شروع فرآیند آموزش ---
    print("Starting training process...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, hyperparams)
    print("Training finished.")

if __name__ == "__main__":
    main()