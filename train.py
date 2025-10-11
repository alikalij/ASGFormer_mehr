# train.py

import torch
from tqdm import tqdm
from utils.utils import load_checkpoint_dynamic, save_checkpoint
from utils.metrics import calculate_metrics

def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, hyperparams):
    """
    تابع اصلی برای آموزش و اعتبارسنجی مدل.
    """
    model, optimizer, start_epoch, train_losses, val_losses = load_checkpoint_dynamic(
        model=model,
        directory=hyperparams['checkpoint_dir'], 
        optimizer=optimizer, 
        for_training=True
    )
    
    device = torch.device(hyperparams['device'])
    model = model.to(device)
    num_classes = hyperparams['num_classes']

    accumulation_steps = 4  # انباشت گرادیان برای شبیه‌سازی بچ سایز بزرگتر
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(start_epoch, hyperparams['num_epochs']):
        # ================== حلقه آموزش ==================
        model.train()
        total_train_loss = 0.0
        
        # 💡 بهبود: متغیرهایی برای agregat کردن متریک‌های آموزش
        train_total_correct = 0
        train_total_points = 0
        train_total_intersection = torch.zeros(num_classes, device=device)
        train_total_union = torch.zeros(num_classes, device=device)
        
        train_loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']} [Train]")
        
        # ✅ اصلاح: استفاده از enumerate برای دریافت `step`
        for step, batch in enumerate(train_loop):
            batch = batch.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs, labels = model(batch)
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
            
            # 💡 بهبود: محاسبه و agregat کردن متریک‌های آموزش
            _, _, batch_intersection, batch_union = calculate_metrics(outputs, labels, num_classes)
            train_total_intersection += batch_intersection
            train_total_union += batch_union
            preds = torch.argmax(outputs, dim=1)
            train_total_correct += (preds == labels).sum().item()
            train_total_points += labels.numel()

            # ✅ اصلاح: نمایش میانگین loss لحظه‌ای
            current_avg_loss = total_train_loss / (step + 1)
            train_loop.set_postfix(loss=f"{current_avg_loss:.4f}")
        
        avg_train_loss = total_train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)
        train_oa = train_total_correct / train_total_points
        train_miou = torch.mean(train_total_intersection / (train_total_union + 1e-8)).item()

        # ================== حلقه اعتبارسنجی ==================
        model.eval()
        total_val_loss = 0.0
        
        val_total_correct = 0
        val_total_points = 0
        val_total_intersection = torch.zeros(num_classes, device=device)
        val_total_union = torch.zeros(num_classes, device=device)

        val_loop = tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']} [Val]")
        with torch.no_grad():
            for batch in val_loop:
                batch = batch.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs, labels = model(batch)
                    loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()

                _, _, batch_intersection, batch_union = calculate_metrics(outputs, labels, num_classes)
                val_total_intersection += batch_intersection
                val_total_union += batch_union
                preds = torch.argmax(outputs, dim=1)
                val_total_correct += (preds == labels).sum().item()
                val_total_points += labels.numel()
        
        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        
        val_oa = val_total_correct / val_total_points if val_total_points > 0 else 0.0
        val_iou_per_class = val_total_intersection / (val_total_union + 1e-8)
        val_miou = torch.mean(val_iou_per_class).item()

        if scheduler:
            scheduler.step()
        
        # 💡 بهبود: چاپ نتایج کامل‌تر در پایان هر Epoch
        print(
            f"\nEpoch {epoch+1} Summary:"
            f"\n  Train -> Loss: {avg_train_loss:.4f}, OA: {train_oa:.4f}, mIoU: {train_miou:.4f}"
            f"\n  Val   -> Loss: {avg_val_loss:.4f}, OA: {val_oa:.4f}, mIoU: {val_miou:.4f}"
            f"\n  LR: {scheduler.get_last_lr()[0]:.6f}\n"
        )

        save_checkpoint(model, optimizer, epoch + 1, train_losses, val_losses, hyperparams['checkpoint_dir'])