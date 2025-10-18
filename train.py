# train.py
import torch
from tqdm import tqdm
import os # Needed for os.path.join

# Import project modules
from utils.utils import load_checkpoint_dynamic, save_checkpoint
from utils.metrics import calculate_metrics

def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, hyperparams):
    """
    تابع اصلی برای آموزش و اعتبارسنجی مدل با پشتیبانی از AMP و انباشت گرادیان.
    """
    device = torch.device(hyperparams['device'])
    num_classes = hyperparams['num_classes']
    checkpoint_dir = hyperparams['checkpoint_dir']

    # بارگذاری checkpoint (در صورت وجود)
    model, optimizer, start_epoch, train_losses, val_losses = load_checkpoint_dynamic(
        model=model,
        directory=checkpoint_dir,
        optimizer=optimizer,
        for_training=True
    )
    model = model.to(device) # Ensure model is on the correct device after loading

    # انباشت گرادیان (Gradient Accumulation)
    accumulation_steps = 4 # Process 4 batches before each optimizer step
    print(f"Using Gradient Accumulation with {accumulation_steps} steps.")

    # Automatic Mixed Precision (AMP) Scaler
    # Enabled only if device is CUDA
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Automatic Mixed Precision (AMP) enabled: {use_amp}")

    print(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, hyperparams['num_epochs']):
        epoch_num = epoch + 1 # Use 1-based epoch number for logging

        # ================== حلقه آموزش ==================
        model.train()
        total_train_loss = 0.0
        train_total_correct = 0
        train_total_points = 0
        train_total_intersection = torch.zeros(num_classes, device=device, dtype=torch.long)
        train_total_union = torch.zeros(num_classes, device=device, dtype=torch.long)

        # Initialize optimizer gradients only at the beginning of the epoch
        optimizer.zero_grad()

        train_loop = tqdm(dataloader_train, desc=f"Epoch {epoch_num}/{hyperparams['num_epochs']} [Train]")
        for step, batch in enumerate(train_loop):
            batch = batch.to(device)

            # AMP context manager
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Ensure model gets the whole batch object
                outputs, labels = model(batch)
                loss = criterion(outputs, labels)

                # Normalize loss for accumulation
                loss = loss / accumulation_steps

            # Scale loss and backward pass
            scaler.scale(loss).backward()

            # Optimizer step and gradient zeroing after accumulation_steps
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader_train):
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                scaler.step(optimizer) # Optimizer step
                scaler.update() # Update scaler for next iteration
                optimizer.zero_grad() # Zero gradients for the next accumulation cycle

            # --- Update Metrics ---
            with torch.no_grad(): # Ensure metric calculation doesn't affect gradients
                total_train_loss += loss.item() * accumulation_steps # Use original loss scale for logging
                _, _, batch_intersection, batch_union = calculate_metrics(outputs, labels, num_classes)
                train_total_intersection += batch_intersection
                train_total_union += batch_union
                preds = torch.argmax(outputs, dim=1)
                train_total_correct += (preds == labels).sum().item()
                train_total_points += labels.numel()

            # Update progress bar description
            current_avg_loss = total_train_loss / (step + 1)
            train_loop.set_postfix(loss=f"{current_avg_loss:.4f}")

        # --- Calculate final training metrics for the epoch ---
        avg_train_loss = total_train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)
        train_oa = train_total_correct / train_total_points if train_total_points > 0 else 0.0
        # Calculate mIoU safely, avoiding division by zero for classes not present
        train_iou_per_class = train_total_intersection.float() / (train_total_union.float() + 1e-8)
        train_miou = torch.mean(train_iou_per_class[train_total_union > 0]).item() # Only average valid classes

        # ================== حلقه اعتبارسنجی ==================
        model.eval()
        total_val_loss = 0.0
        val_total_correct = 0
        val_total_points = 0
        val_total_intersection = torch.zeros(num_classes, device=device, dtype=torch.long)
        val_total_union = torch.zeros(num_classes, device=device, dtype=torch.long)

        val_loop = tqdm(dataloader_val, desc=f"Epoch {epoch_num}/{hyperparams['num_epochs']} [Val]")
        with torch.no_grad(): # No gradients needed for validation
            for batch in val_loop:
                batch = batch.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs, labels = model(batch)
                    loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                _, _, batch_intersection, batch_union = calculate_metrics(outputs, labels, num_classes)
                val_total_intersection += batch_intersection
                val_total_union += batch_union
                preds = torch.argmax(outputs, dim=1)
                val_total_correct += (preds == labels).sum().item()
                val_total_points += labels.numel()

                # Update validation progress bar description
                current_avg_val_loss = total_val_loss / (val_loop.n + 1) # val_loop.n is number of iterations done
                val_loop.set_postfix(loss=f"{current_avg_val_loss:.4f}")

        # --- Calculate final validation metrics for the epoch ---
        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)

        val_oa = val_total_correct / val_total_points if val_total_points > 0 else 0.0
        val_iou_per_class = val_total_intersection.float() / (val_total_union.float() + 1e-8)
        # Calculate mIoU only over classes present in the validation set union
        valid_classes_mask = val_total_union > 0
        if valid_classes_mask.any():
            val_miou = torch.mean(val_iou_per_class[valid_classes_mask]).item()
        else:
            val_miou = 0.0 # Handle case where validation set might be empty or problematic

        # --- Step the scheduler ---
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr'] # Get LR directly if no scheduler


        # --- Print Epoch Summary ---
        print(
            f"\nEpoch {epoch_num} Summary:"
            f"\n  Train -> Loss: {avg_train_loss:.4f}, OA: {train_oa:.4f}, mIoU: {train_miou:.4f}"
            f"\n  Val   -> Loss: {avg_val_loss:.4f}, OA: {val_oa:.4f}, mIoU: {val_miou:.4f}"
            f"\n  LR: {current_lr:.6f}\n"
        )

        # --- Save Checkpoint ---
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_num, # Save with 1-based epoch number
            train_losses=train_losses,
            val_losses=val_losses,
            base_dir=checkpoint_dir
        )
    print("Training loop finished.")