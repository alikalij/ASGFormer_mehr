# data/dataset.py

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def read_file_list(file_path):
    """خواندن لیست مسیر فایل‌ها از یک فایل متنی."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def compute_class_weights(labels_list, num_classes):
    """محاسبه وزن کلاس‌ها برای مقابله با عدم تعادل."""
    total_labels = np.concatenate(labels_list)
    class_counts = np.bincount(total_labels, minlength=num_classes)
    
    # جلوگیری از تقسیم بر صفر برای کلاس‌هایی که وجود ندارند
    total_samples = class_counts.sum()
    if total_samples == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    return torch.tensor(class_weights, dtype=torch.float32)

class PointCloudProcessor:
    """کلاسی برای پردازش و نرمال‌سازی داده‌های ابر نقاط."""
    def __init__(self, num_points):
        self.num_points = num_points

    def _normalize_points(self, points):
        """نرمال‌سازی ویژگی‌ها به صورت استاندارد."""
        # مرکزیت XYZ بر اساس میانگین
        points[:, :3] = points[:, :3] - np.mean(points[:, :3], axis=0)
        # نرمال‌سازی رنگ‌ها به بازه [0, 1]
        points[:, 3:6] = points[:, 3:6] / 255.0
        # بردارهای نرمال معمولاً از قبل واحد هستند و نیازی به نرمال‌سازی ندارند
        return points

    def process(self, data, labels):
        """نمونه‌برداری و نرمال‌سازی داده‌ها."""
        points = self._normalize_points(data)
        
        # نمونه‌برداری تصادفی برای رسیدن به تعداد نقاط ثابت
        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            # اگر تعداد نقاط کمتر بود، با تکرار به تعداد مورد نظر می‌رسانیم
            choice = np.random.choice(len(points), self.num_points, replace=True)
            
        points = points[choice, :]
        labels = labels[choice]
        
        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()

class H5Dataset(Dataset):
    """کلاس دیتاست برای خواندن فایل‌های H5."""
    def __init__(self, file_paths, processor, base_path):
        self.file_paths = [os.path.join(base_path, f) for f in file_paths]
        self.processor = processor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        with h5py.File(h5_path, 'r') as f:
            # ✅ اصلاح: تمام ۹ ستون خوانده می‌شود
            data = f['data'][:]
            labels = f['label'][:].flatten()

        features, labels = self.processor.process(data, labels)
        
        # ✅ اصلاح: تفکیک pos از x
        # pos فقط شامل XYZ است
        pos = features[:, :3]
        # x شامل تمام ۹ ویژگی است
        x = features
        
        return Data(x=x, pos=pos, y=labels)