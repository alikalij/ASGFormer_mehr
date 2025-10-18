import os
import sys
from pathlib import Path
import torch

# تشخیص خودکار محیط اجرا
#IS_COLAB = 'google.colab' in sys.modules
IS_COLAB = os.getenv('COLAB_ENV') == '1'
IS_LOCAL = not IS_COLAB

#print("IS_COLAB =>", IS_COLAB)

# تنظیمات پایه
BASE_CONFIG = {
    # پارامترهای مشترک
    'main_output_dim': 13,
    'batch_size': 2,
    'num_epochs': 40,
    'learning_rate': 1e-3,
    'knn_param': 4,
    'num_points': 4096,
    'dropout_param': 0.1,
    'weight_decay': 0.05, #1e-3,
    # تنظیمات دستگاه
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# تنظیمات خاص محیط
ENV_SPECIFIC = {
    'colab': {
        'dataset_path': '/content/dataset/s3dis',
        'checkpoint_dir': '/content/drive/MyDrive/saved_models/',
        'requirements': True
    },
    'local': {
        'dataset_path': str(Path(__file__).parent.parent / 'data/s3dis'),
        'checkpoint_dir': str(Path(__file__).parent.parent / 'saved_models/'),
        'requirements': False
    }
}

# ترکیب تنظیمات
CONFIG = {**BASE_CONFIG, **ENV_SPECIFIC['colab' if IS_COLAB else 'local']}