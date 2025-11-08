# evaluator.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os

from models.model import ASGFormer
from data.dataset import H5Dataset, PointCloudProcessor, read_file_list
from torch_geometric.loader import DataLoader
from utils.utils import load_checkpoint_dynamic
from utils.metrics import calculate_final_metrics # تابع جدید برای نتایج نهایی

# نقشه رنگ S3DIS (مانند قبل)
S3DIS_COLOR_MAP = np.array([
    [152, 223, 138],  # 0: ceiling
    [174, 199, 232],  # 1: floor
    [31, 119, 180],   # 2: wall
    [255, 187, 120],  # 3: beam
    [188, 189, 34],   # 4: column
    [140, 86, 75],    # 5: window
    [255, 152, 150],  # 6: door
    [214, 39, 40],    # 7: table
    [197, 176, 213],  # 8: chair
    [148, 103, 189],  # 9: sofa
    [196, 156, 148],  # 10: bookcase
    [23, 190, 207],   # 11: board
    [247, 182, 210],  # 12: clutter
    # اگر کلاس‌های بیشتری دارید، رنگ‌های بیشتری اضافه کنید
]) / 255.0  # نرمال‌سازی به بازه [0, 1]

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.checkpoint_dir = config['checkpoint_dir']
        self.num_classes = config['num_classes']
        
        # ساخت مدل (فقط برای بارگذاری checkpoint لازم است)
        self.model = ASGFormer(
            feature_dim=config['feature_dim'],
            main_input_dim=config['main_input_dim'],
            main_output_dim=self.num_classes,
            stages_config=config['stages_config'],
            knn_param=config['knn_param'],
            dropout_param=config['dropout_param']
        )
        
        # بارگذاری آخرین checkpoint
        self._load_latest_checkpoint()

    def _load_latest_checkpoint(self):
        print("در حال بارگذاری آخرین checkpoint برای ارزیابی...")
        self.model, _, self.epoch, self.train_losses, self.val_losses = load_checkpoint_dynamic(
            self.model, self.checkpoint_dir, optimizer=None, for_training=False
        )
        self.model.to(self.device)
        self.model.eval() # مدل همیشه در حالت ارزیابی است

    def plot_loss(self, save_path="loss_curve.png"):
        if not self.train_losses or not self.val_losses:
            print("تاریخچه Loss یافت نشد.")
            return

        plt.figure(figsize=(12, 7))
        plt.plot(self.train_losses, label="Train Loss", marker='o', linestyle='-')
        plt.plot(self.val_losses, label="Validation Loss", marker='s', linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss up to Epoch {self.epoch}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"نمودار Loss در فایل '{save_path}' ذخیره شد.")
        plt.show()

    def test(self, dataloader):
        print("شروع ارزیابی نهایی مدل...")
        all_preds = []
        all_labels = []

        with torch.no_grad():
            test_loop = tqdm(dataloader, desc="Testing")
            for data in test_loop:
                data = data.to(self.device)
                outputs, labels = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        overall_acc, mIoU, iou_per_class = calculate_final_metrics(all_preds, all_labels, self.num_classes)
        
        print("\n--- Final Test Results ---")
        print(f"Checkpoint Epoch: {self.epoch}")
        print(f"Overall Accuracy (OA): {overall_acc * 100:.2f}%")
        print(f"Mean IoU (mIoU): {mIoU * 100:.2f}%")
        print("\nIoU per class:")
        for i, iou in enumerate(iou_per_class):
            print(f"  Class {i}: {iou * 100:.2f}%")
        print("--------------------------")
        return overall_acc, mIoU, iou_per_class

    def visualize(self, dataloader, num_samples=3):
        print(f"در حال آماده‌سازی برای بصری‌سازی {num_samples} نمونه...")
        samples_processed = 0
        # 💡 بهبود: استفاده از try-except برای مدیریت خطای احتمالی Open3D
        try:
            with torch.no_grad():
                for data in dataloader:
                    if samples_processed >= num_samples: break
                    
                    data = data.to(self.device)
                    # 💡 نکته: مطمئن شوید مدل در حالت eval است (در __init__ انجام شده)
                    outputs, labels = self.model(data) 
                    preds = torch.argmax(outputs, dim=1)

                    points = data.pos.cpu().numpy()
                    true_labels = labels.cpu().numpy()
                    pred_labels = preds.cpu().numpy()

                    # --- ساخت ابر نقاط Open3D ---
                    
                    # ابر نقاط Ground Truth (واقعی)
                    gt_pcd = o3d.geometry.PointCloud()
                    gt_pcd.points = o3d.utility.Vector3dVector(points)
                    # ✅ تکمیل شده: تنظیم رنگ‌ها بر اساس برچسب واقعی
                    # جلوگیری از خطای ایندکس اگر برچسبی خارج از محدوده باشد
                    valid_gt_labels = np.clip(true_labels, 0, S3DIS_COLOR_MAP.shape[0] - 1)
                    gt_pcd.colors = o3d.utility.Vector3dVector(S3DIS_COLOR_MAP[valid_gt_labels])

                    # ابر نقاط Prediction (پیش‌بینی مدل)
                    pred_pcd = o3d.geometry.PointCloud()
                    pred_pcd.points = o3d.utility.Vector3dVector(points)
                    # ✅ تکمیل شده: تنظیم رنگ‌ها بر اساس پیش‌بینی مدل
                    valid_pred_labels = np.clip(pred_labels, 0, S3DIS_COLOR_MAP.shape[0] - 1)
                    pred_pcd.colors = o3d.utility.Vector3dVector(S3DIS_COLOR_MAP[valid_pred_labels])
                    
                    # انتقال ابر نقاط پیش‌بینی شده به کنار ابر نقاط واقعی
                    translation_vector = np.array([(np.max(points[:, 0]) - np.min(points[:, 0])) * 1.1, 0, 0])
                    pred_pcd.translate(translation_vector)

                    # --- نمایش ---
                    print(f"\nنمایش نمونه {samples_processed + 1} (پنجره Open3D باز می‌شود):")
                    print("  ابر نقاط سمت چپ: Ground Truth (واقعی)")
                    print("  ابر نقاط سمت راست: Prediction (پیش‌بینی مدل)")
                    o3d.visualization.draw_geometries(
                        [gt_pcd, pred_pcd], 
                        window_name=f"Sample {samples_processed + 1} | Left: GT, Right: Pred",
                        width=1280, height=720
                    )
                    
                    samples_processed += 1
        except ImportError:
             print("\nخطا: کتابخانه open3d نصب نشده یا به درستی import نشده است.")
             print("برای استفاده از بصری‌سازی، لطفاً open3d را نصب کنید: pip install open3d")
        except Exception as e:
            print(f"\nخطا در هنگام بصری‌سازی: {e}")