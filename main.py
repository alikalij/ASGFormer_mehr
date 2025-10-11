# main.py

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models.model import ASGFormer
# ✅ اصلاح: وارد کردن کلاس‌های جدید دیتاست
from data.dataset import H5Dataset, PointCloudProcessor, read_file_list, compute_class_weights
from train import train_model
from configs.env_config import CONFIG
import os
import h5py

def main():
    main_output_dim = 13
    hyperparams = {
        'device': CONFIG['device'],
        'learning_rate': CONFIG['learning_rate'],
        'batch_size': CONFIG['batch_size'],
        'num_epochs': CONFIG['num_epochs'],
        'dataset_path': CONFIG['dataset_path'],
        'knn_param': CONFIG['knn_param'],
        'num_points': CONFIG['num_points'],
        'dropout_param': CONFIG['dropout_param'],
        'weight_decay': CONFIG['weight_decay'],
        'checkpoint_dir': CONFIG['checkpoint_dir'],
        'num_classes': main_output_dim
    }

    # ✅ بهبود: downsample_ratio: 1.0 به None تغییر کرد تا واضح‌تر باشد
    stages_config = [
        {'hidden_dim': 32, 'num_layers': 1, 'downsample_ratio': None},
        {'hidden_dim': 64, 'num_layers': 2, 'downsample_ratio': 0.25},
        {'hidden_dim': 128, 'num_layers': 4, 'downsample_ratio': 0.25},
        {'hidden_dim': 256, 'num_layers': 2, 'downsample_ratio': 0.25},
        {'hidden_dim': 512, 'num_layers': 2, 'downsample_ratio': 0.25},
    ]

    dataset_path = hyperparams['dataset_path']
    train_files = read_file_list(os.path.join(dataset_path, "list", "train5.txt"))
    val_files = read_file_list(os.path.join(dataset_path, "list", "val5.txt"))

    # ✅ اصلاح: استفاده از PointCloudProcessor جدید
    processor = PointCloudProcessor(num_points=hyperparams['num_points'])
    train_dataset = H5Dataset(train_files, processor, dataset_path)
    val_dataset = H5Dataset(val_files, processor, dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=4)

    # ✅ اصلاح: feature_dim به ۹ تغییر یافت
    feature_dim = 9
    main_input_dim = 32 
    main_output_dim = 13 # تعداد کلاس‌های S3DIS

    model = ASGFormer(feature_dim=feature_dim,
                      main_input_dim=main_input_dim,
                      main_output_dim=main_output_dim,
                      stages_config=stages_config,
                      knn_param=hyperparams['knn_param'],
                      dropout_param=hyperparams['dropout_param']).to(hyperparams['device'])

    # ✅ اصلاح: محاسبه وزن‌ها با استفاده از تابع جدید
    all_train_labels = [h5py.File(os.path.join(dataset_path, f), 'r')['label'][:] for f in train_files]
    class_weights = compute_class_weights(all_train_labels, num_classes=main_output_dim).to(hyperparams['device'])
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(...)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(...)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, hyperparams)

if __name__ == "__main__":
    main()