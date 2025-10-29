# quantize.py

import torch
import torch.quantization
import argparse
import os
from pathlib import Path

from models.model import ASGFormer
from configs.env_config import CONFIG # برای بارگذاری معماری مدل
from utils.utils import load_checkpoint_dynamic # برای بارگذاری مدل آموزش‌دیده

def quantize_model(model, quantization_type='dynamic'):
    """
    مدل ورودی را با روش مشخص شده کوانتیزه می‌کند.
    """
    model.to('cpu') # کوانتیزاسیون معمولا روی CPU انجام و استفاده می‌شود
    model.eval()

    if quantization_type == 'dynamic':
        print("Applying Post Training Dynamic Quantization...")
        # فقط لایه‌های Linear کوانتیزه می‌شوند (رایج‌ترین حالت)
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Dynamic Quantization complete.")
    # elif quantization_type == 'static':
    #     # Static Quantization نیاز به داده کالیبراسیون دارد و پیچیده‌تر است
    #     print("Applying Post Training Static Quantization...")
    #     # ... (کد مربوط به کوانتیزاسیون استاتیک) ...
    #     print("Static Quantization complete.")
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

    return quantized_model

def save_quantized_model(quantized_model, original_checkpoint_dir, filename="quantized_model.pth"):
    """مدل کوانتیزه شده را ذخیره می‌کند."""
    save_dir = Path(original_checkpoint_dir) / "quantized"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / filename
    
    # برای مدل‌های کوانتیزه شده، state_dict را با torch.save ذخیره می‌کنیم
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantized model saved to: {save_path}")
    return save_path

def measure_size(model_path):
    """اندازه فایل مدل را بر حسب مگابایت برمی‌گرداند."""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a trained ASGFormer model.")
    # می‌توان نوع کوانتیزاسیون را نیز به عنوان آرگومان اضافه کرد
    # parser.add_argument('--type', type=str, default='dynamic', choices=['dynamic', 'static'], help="Quantization type.")
    args = parser.parse_args()

    # --- بارگذاری مدل آموزش‌دیده ---
    hyperparams = CONFIG
    
    # 1. ساخت مدل با معماری اصلی
    original_model = ASGFormer(
        feature_dim=hyperparams['feature_dim'],
        main_input_dim=hyperparams['main_input_dim'],
        main_output_dim=hyperparams['num_classes'],
        stages_config=hyperparams['stages_config'],
        knn_param=hyperparams['knn_param'],
        dropout_param=hyperparams['dropout_param']
    )

    # 2. بارگذاری وزن‌های آموزش‌دیده از آخرین checkpoint
    print("Loading the latest trained checkpoint...")
    # از تابع load_checkpoint_dynamic استفاده می‌کنیم اما optimizer را نمی‌خواهیم
    loaded_model, _, loaded_epoch, _, _ = load_checkpoint_dynamic(
        original_model, directory=hyperparams['checkpoint_dir'], for_training=False
    )
    if loaded_epoch == 0:
         print("Error: No trained checkpoint found. Please train the model first.")
         exit()
         
    print(f"Loaded model trained up to epoch {loaded_epoch}.")
    loaded_model.eval() # اطمینان از اینکه مدل در حالت ارزیابی است

    # --- انجام کوانتیزاسیون ---
    quantized_model = quantize_model(loaded_model, quantization_type='dynamic')

    # --- ذخیره مدل کوانتیزه شده ---
    quantized_model_path = save_quantized_model(quantized_model, hyperparams['checkpoint_dir'])

    # --- مقایسه اندازه مدل‌ها ---
    try:
         # یافتن مسیر checkpoint اصلی برای مقایسه اندازه
         from utils.utils import find_latest_checkpoint
         original_checkpoint_path = find_latest_checkpoint(hyperparams['checkpoint_dir'])
         
         if original_checkpoint_path:
              original_size = measure_size(original_checkpoint_path)
              quantized_size = measure_size(quantized_model_path)
              print("\n--- Model Size Comparison ---")
              print(f"Original model size: {original_size:.2f} MB")
              print(f"Quantized model size: {quantized_size:.2f} MB")
              print(f"Reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
         else:
              print("\nCould not find original checkpoint to compare size.")
              
    except Exception as e:
         print(f"Could not compare model sizes: {e}")
         
    print("\nQuantization process finished.")