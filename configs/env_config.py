# configs/env_config.py
import torch
import os
from pathlib import Path # Import Path for cleaner local path handling

# --- تشخیص خودکار محیط ---
# Check for Colab environment variable, safer than checking sys.modules
IS_COLAB = 'COLAB_GPU' in os.environ or os.getenv('COLAB_ENV') == '1'
print(f"IS_COLAB => {IS_COLAB}")

# --- مسیرهای اصلی ---
# Use Path objects for better cross-platform compatibility
if IS_COLAB:
    BASE_DIR = Path("/content")
    # Mount Google Drive if in Colab
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = Path("/content/drive/MyDrive") # Adjust if your Drive path is different
    # Define project directory within Colab's filesystem
    PROJECT_DIR = BASE_DIR / "ASGFormer_mehr"
else:
    # Assuming the script is run from the project root locally
    PROJECT_DIR = Path(__file__).parent.parent # Goes up two levels from configs/env_config.py to the project root
    DRIVE_DIR = Path("G:/My Drive") # Adjust your local "Google Drive equivalent" path if needed

# --- تنظیمات اصلی پروژه ---
CONFIG = {
    # تنظیمات محیط و دستگاه
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # تنظیمات دیتاست
    'dataset_path': str(PROJECT_DIR / 'data/s3dis-mini'), # Use str() for compatibility with older functions
    'num_points': 4096,
    'num_classes': 13, # S3DIS has 13 classes

    # هایپرپارامترهای آموزش
    'learning_rate': 1e-3,
    'batch_size': 8, # Keep batch size reasonable for Colab memory
    'num_epochs': 40,
    'weight_decay': 1e-4, # Common value for weight decay
    'label_smoothing': 0.1, # Label smoothing factor

    # هایپرپارامترهای مدل
    'feature_dim': 9, # XYZ + RGB + Normals
    'main_input_dim': 32, # Dimension after initial MLPs, must match first hidden_dim
    'knn_param': 16,
    'dropout_param': 0.1,

    # مسیرهای ذخیره‌سازی
    'checkpoint_dir': str(DRIVE_DIR / "saved_models/ASGFormer"), # Save checkpoints to Drive

    # پیکربندی معماری مدل (Encoder stages)
    'stages_config': [
        {'hidden_dim': 32, 'num_layers': 1, 'downsample_ratio': None}, # First stage (MLP only), no downsample
        {'hidden_dim': 64, 'num_layers': 2, 'downsample_ratio': 0.25}, # N -> N/4
        {'hidden_dim': 128, 'num_layers': 4, 'downsample_ratio': 0.25}, # N/4 -> N/16
        {'hidden_dim': 256, 'num_layers': 2, 'downsample_ratio': 0.25}, # N/16 -> N/64
        {'hidden_dim': 512, 'num_layers': 2, 'downsample_ratio': 0.25}, # N/64 -> N/256
    ],

    # تنظیمات Data Loader
    'num_workers': 2 if IS_COLAB else 4, # Colab often has limited CPU resources
    'pin_memory': torch.cuda.is_available() # Enable pin_memory only if using GPU
}

# Ensure checkpoint directory exists
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# Print device info for confirmation
print(f"Using device: {CONFIG['device']}")
print(f"Dataset path: {CONFIG['dataset_path']}")
print(f"Checkpoint directory: {CONFIG['checkpoint_dir']}")