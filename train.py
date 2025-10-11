# train.py

import torch
import numpy as np
from tqdm import tqdm
from utils.utils import load_checkpoint_dynamic, save_checkpoint
# ✅ وارد کردن تابع جدید برای محاسبه متریک‌ها
from utils.metrics import calculate_metrics 

def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, hyperparams):
    model, optimizer, start_epoch, train_losses, val_losses = load_checkpoint_dynamic(
        model=model,
        directory=hyperparams['checkpoint_dir'], 
        optimizer=optimizer, 
        for_training=True
    )
    
    device = torch.device(hyperparams['device'])
    model = model.to(device)
    num_classes = hyperparams['num_classes'] # تعداد کلاس‌ها را از هایپرپارامترها می‌گیریم

    accumulation_steps = 4
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(start_epoch, hyperparams['num_epochs']):
        # --- حلقه آموزش ---
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']} [Train]")
        
        for step, batch in enumerate(train_loop):
            x, pos, labels, batch_tensor  = batch.x.to(device), batch.pos.to(device), batch.y.to(device), batch.batch.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs, _ = model(x, pos, labels, batch_tensor)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader_train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * accumulation_steps
            train_loop.set_postfix(loss=total_train_loss / (step + 1))
        
        avg_train_loss = total_train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)

        # --- حلقه اعتبارسنجی با محاسبه متریک‌ها ---
        model.eval()
        total_val_loss = 0.0
        # ✅ متغیرهایی برای agregat کردن متریک‌ها در تمام بچ‌ها
        total_correct_points = 0
        total_points_in_val = 0
        total_intersection = torch.zeros(num_classes, device=device)
        total_union = torch.zeros(num_classes, device=device)

        val_loop = tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']} [Val]")
        with torch.no_grad():
            for batch in val_loop:
                x, pos, labels, batch_tensor = batch.x.to(device), batch.pos.to(device), batch.y.to(device), batch.batch.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs, _ = model(x, pos, labels, batch_tensor)
                    loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()

                # ✅ محاسبه متریک‌ها برای بچ فعلی و agregat کردن نتایج
                _, _, batch_intersection, batch_union = calculate_metrics(outputs, labels, num_classes)
                total_intersection += batch_intersection
                total_union += batch_union

                preds = torch.argmax(outputs, dim=1)
                total_correct_points += (preds == labels).sum().item()
                total_points_in_val += labels.numel()

        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        
        # ✅ محاسبه متریک‌های نهایی برای کل دیتاست اعتبارسنجی
        val_oa = total_correct_points / total_points_in_val if total_points_in_val > 0 else 0.0
        val_iou = total_intersection / (total_union + 1e-8)
        val_miou = torch.mean(val_iou).item() # mIoU: میانگین IoU در تمام کلاس‌ها

        if scheduler:
            scheduler.step()
        
        # ✅ چاپ نتایج شامل متریک‌های جدید
        print(f"Epoch {epoch+1} Summary: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Val OA={val_oa:.4f}, Val mIoU={val_miou:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

        save_checkpoint(model, optimizer, epoch + 1, train_losses, val_losses, hyperparams['checkpoint_dir'])