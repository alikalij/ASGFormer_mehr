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

# data/dataset.py

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# ... (توابع read_file_list و compute_class_weights بدون تغییر باقی می‌مانند) ...

class PointCloudProcessor:
    """کلاسی برای پردازش و نرمال‌سازی داده‌های ابر نقاط."""
    def __init__(self, num_points):
        self.num_points = num_points

    def _normalize_points(self, points_features):
        """
        نرمال‌سازی ویژگی‌ها:
        1. مرکزیت XYZ با کم کردن میانگین.
        2. مقیاس‌دهی XYZ برای قرارگیری در کره واحد.
        3. نرمال‌سازی رنگ RGB به [0, 1].
        """
        # --- نرمال‌سازی XYZ ---
        xyz = points_features[:, :3]

        # 1. مرکزیت (Centering)
        centroid = np.mean(xyz, axis=0)
        xyz_centered = xyz - centroid

        # 2. مقیاس‌دهی به کره واحد (Scaling to Unit Sphere)
        # محاسبه فاصله اقلیدسی هر نقطه از مبدأ جدید (0,0,0)
        distances = np.sqrt(np.sum(xyz_centered**2, axis=1))
        # یافتن حداکثر فاصله (شعاع)
        max_distance = np.max(distances)
        # جلوگیری از تقسیم بر صفر اگر همه نقاط در یک نقطه باشند
        if max_distance > 1e-6:
             xyz_normalized = xyz_centered / max_distance
        else:
             xyz_normalized = xyz_centered # در این حالت نیازی به مقیاس‌دهی نیست

        # --- نرمال‌سازی RGB ---
        rgb = points_features[:, 3:6]
        rgb_normalized = rgb / 255.0

        # --- نرمال‌ها (Normals) ---
        # فرض می‌کنیم نرمال‌ها در ستون‌های 6, 7, 8 هستند و از قبل واحد هستند
        normals = points_features[:, 6:9]

        # ترکیب مجدد ویژگی‌های نرمال‌شده
        normalized_features = np.hstack((xyz_normalized, rgb_normalized, normals))

        return normalized_features

    def process(self, data, labels):
        """نمونه‌برداری و نرمال‌سازی داده‌ها."""
        # 💡 نکته: نرمال‌سازی قبل از نمونه‌برداری انجام می‌شود
        # تا مرکزیت و مقیاس بر اساس کل ابر نقاط اصلی محاسبه شود.
        normalized_features = self._normalize_points(data)

        # نمونه‌برداری تصادفی برای رسیدن به تعداد نقاط ثابت
        num_original_points = len(normalized_features)
        if num_original_points > self.num_points:
            choice = np.random.choice(num_original_points, self.num_points, replace=False)
        else:
            # اگر تعداد نقاط کمتر بود، با تکرار به تعداد مورد نظر می‌رسانیم
            choice = np.random.choice(num_original_points, self.num_points, replace=True)

        sampled_features = normalized_features[choice, :]
        sampled_labels = labels[choice]

        return torch.from_numpy(sampled_features).float(), torch.from_numpy(sampled_labels).long()



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