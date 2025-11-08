# data/dataset.py

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import random 

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def compute_class_weights(labels_list, num_classes):
    total_labels = np.concatenate(labels_list)
    class_counts = np.bincount(total_labels, minlength=num_classes)
    
    total_samples = class_counts.sum()
    if total_samples == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    return torch.tensor(class_weights, dtype=torch.float32)


class PointCloudProcessor:
    def __init__(self, num_points, is_training=True):
        self.num_points = num_points
        self.is_training = is_training 

    def _normalize_points(self, points_features):
        xyz = points_features[:, :3]

        centroid = np.mean(xyz, axis=0)
        xyz_centered = xyz - centroid

        distances = np.sqrt(np.sum(xyz_centered**2, axis=1))
        max_distance = np.max(distances)
        if max_distance > 1e-6:
             xyz_normalized = xyz_centered / max_distance
        else:
             xyz_normalized = xyz_centered 

        rgb = points_features[:, 3:6]
        rgb_normalized = rgb / 255.0

        normals = points_features[:, 6:9]

        normalized_z = xyz_normalized[:, 2:3]

        normalized_features = np.hstack((xyz_normalized, rgb_normalized, normals))

        return normalized_features 

    def _apply_augmentation(self, points_features):
        scale = np.random.uniform(0.9, 1.1)
        points_features[:, :3] *= scale

        if np.random.rand() < 0.2: 
             points_features[:, 3:6] = 0.0 
             
        jitter = np.random.normal(0, 0.02, (points_features.shape[0], 3))
        points_features[:, :3] += jitter

        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        points_features[:, :3] = points_features[:, :3] @ rotation_matrix.T
        points_features[:, 6:9] = points_features[:, 6:9] @ rotation_matrix.T
        
        return points_features

    def process(self, data, labels):
        num_original_points = len(data)
        if num_original_points > self.num_points:
            choice = np.random.choice(num_original_points, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_original_points, self.num_points, replace=True)
            
        sampled_data = data[choice, :]
        sampled_labels = labels[choice]

        if self.is_training:
            sampled_data = self._apply_augmentation(sampled_data.copy())

        normalized_features = self._normalize_points(sampled_data)
                
        return torch.from_numpy(normalized_features).float(), torch.from_numpy(sampled_labels).long()


class H5Dataset(Dataset):
    def __init__(self, file_paths, processor, base_path):
        self.file_paths = [os.path.join(base_path, f) for f in file_paths]
        self.processor = processor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
            labels = f['label'][:].flatten()

        features, labels = self.processor.process(data, labels)
        
        pos = features[:, :3]
        x = features[:, 3:] 
        
        return Data(x=x, pos=pos, y=labels)