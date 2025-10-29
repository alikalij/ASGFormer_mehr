import os
import re # برای عبارت منظم
import torch
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, base_dir, filename_prefix="agtransformer"):
    """ذخیره‌ی checkpoint مدل همراه با مدیریت نسخه."""
    # ✅ بهبود: ایجاد دایرکتوری به شکل صحیح
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ✅ بهبود: حذف خط اضافی و ساخت مسیر به صورت استاندارد
    checkpoint_path = os.path.join(base_dir, f"{filename_prefix}_epoch{epoch}_{timestamp}.pth")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': list(train_losses),
        'val_losses': list(val_losses),
    }
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"مدل در epoch {epoch} ذخیره شد: {checkpoint_path}")
    except Exception as e:
        print(f"خطا در ذخیره checkpoint: {e}")

def load_checkpoint(model, checkpoint_path, optimizer=None, for_training=False):
    """بارگذاری checkpoint مدل با مدیریت خطا."""
    if not os.path.exists(checkpoint_path):
        print(f"فایل checkpoint '{checkpoint_path}' یافت نشد. آموزش از ابتدا آغاز می‌شود.")
        return model, optimizer, 0, [], []
    try:        
        # ✅ بهترین تمرین: بارگذاری روی CPU برای جلوگیری از خطای حافظه GPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # نکته: strict=True برای ادامه آموزش بهتر است تا مطمئن شوید معماری تغییر نکرده
        model.load_state_dict(checkpoint['model_state_dict'], strict=True) 
        start_epoch = checkpoint.get('epoch', 0)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        if for_training and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])            
        
        print(f"مدل از epoch {start_epoch} بارگذاری شد. آخرین Loss: {train_losses[-1] if train_losses else 'N/A'}")
        return model, optimizer, start_epoch, train_losses, val_losses
    except Exception as e:
        print(f"خطا در بارگذاری checkpoint: {e}")
        return model, optimizer, 0, [], []

def find_latest_checkpoint(directory, prefix="agtransformer"):
    """پیدا کردن آخرین checkpoint در دایرکتوری مشخص شده."""
    if not os.path.isdir(directory):
        return None
    
    # الگوی عبارت منظم برای یافتن (Epoch) و (Timestamp) در نام فایل
    # مثال: agtransformer_epoch(66)_(20251025_091436).pth
    # (P<epoch>...) نام گروهی برای استخراج Epoch است
    pattern = re.compile(rf"^{prefix}_epoch(?P<epoch>\d+)_(?P<timestamp>\d{{8}}_\d{{6}}).pth$")
    
    checkpoint_data = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # استخراج عدد Epoch و Timestamp به صورت جداگانه
            epoch_num = int(match.group('epoch'))
            timestamp_str = match.group('timestamp')
            
            checkpoint_data.append({
                'filename': filename,
                'epoch': epoch_num,
                'timestamp': timestamp_str,
            })
            
    if not checkpoint_data:
        return None
    
    # مرتب‌سازی:
    # 1. اولویت با 'epoch' (بزرگترین Epoch)
    # 2. در صورت مساوی بودن Epoch، اولویت با 'timestamp' (جدیدترین زمان)
    latest_checkpoint = max(
        checkpoint_data, 
        key=lambda x: (x['epoch'], x['timestamp'])
    )
    
    return os.path.join(directory, latest_checkpoint['filename'])

def load_checkpoint_dynamic(model, directory, optimizer=None, for_training=False):
    checkpoint_path = find_latest_checkpoint(directory)
    if checkpoint_path is None:
        print("هیچ checkpointی در دایرکتوری یافت نشد. آموزش از ابتدا آغاز می‌شود.")
        return model, optimizer, 0, [], []
    return load_checkpoint(model, checkpoint_path, optimizer, for_training)

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN!")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf!")