# configs/env_config.py

import torch
import os
from pathlib import Path # 💡 بهبود: استفاده از pathlib برای مدیریت بهتر مسیرها

# تشخیص خودکار محیط اجرا
IS_COLAB = os.getenv('COLAB_ENV') == '1'
print(f"Running in {'Colab' if IS_COLAB else 'Local'} environment.")

# --- مسیرهای اصلی ---
# استفاده از Path برای سازگاری بهتر بین سیستم‌عامل‌ها
PROJECT_ROOT = Path(__file__).parent.parent # مسیر ریشه پروژه
BASE_DIR = Path("/content") if IS_COLAB else PROJECT_ROOT
DRIVE_DIR = Path("/content/drive/MyDrive") if IS_COLAB else PROJECT_ROOT

# --- تنظیمات اصلی پروژه ---
CONFIG = {
    # تنظیمات محیط و دستگاه
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # تنظیمات دیتاست
    'dataset_path': BASE_DIR / 'datasets' / 's3dis' / 's3dis-mini', # مسیر دیتاست با pathlib
    'num_points': 4096,
    'num_classes': 13,
    'feature_dim': 9, # XYZRGB + Normals
    
    # هایپرپارامترهای آموزش
    'learning_rate': 1e-3, 
    'batch_size': 4,       
    'num_epochs': 120,
    'weight_decay': 1e-4, # مقدار رایج‌تر برای AdamW
    'accumulation_steps': 4, # انباشت گرادیان
    
    # هایپرپارامترهای مدل
    'main_input_dim': 32, # بعد فضای эмبدینگ اولیه
    'knn_param': 16,       
    'dropout_param': 0.1,
    
    # مسیرهای ذخیره‌سازی
    'checkpoint_dir': DRIVE_DIR / "saved_models" , # مسیر checkpoint با pathlib

    # پیکربندی معماری مدل
    'stages_config': [
        {'hidden_dim': 32, 'num_layers': 1, 'downsample_ratio': None},
        {'hidden_dim': 64, 'num_layers': 2, 'downsample_ratio': 0.25},
        {'hidden_dim': 128, 'num_layers': 4, 'downsample_ratio': 0.25},
        {'hidden_dim': 256, 'num_layers': 2, 'downsample_ratio': 0.25},
        {'hidden_dim': 512, 'num_layers': 2, 'downsample_ratio': 0.25},
    ]
}

# اطمینان از وجود دایرکتوری checkpoint
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)