# trainer.py

import torch
from tqdm import tqdm
from utils.utils import load_checkpoint_dynamic, save_checkpoint
from utils.metrics import calculate_metrics 
from utils.losses import LovaszSoftmaxLoss 
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion_ce = criterion
        self.criterion_lovasz = LovaszSoftmaxLoss().to(self.device) 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = config['num_classes']
        self.accumulation_steps = config.get('accumulation_steps', 1) 
        
        self.finetune_epoch = config.get('finetune_epoch', 100) 
        self.finetune_lr = config.get('finetune_lr', 1e-5)

        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []

        self._load_checkpoint()
        
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))

    def _load_checkpoint(self):
        print("در حال تلاش برای بارگذاری آخرین checkpoint...")
        self.model, self.optimizer, self.start_epoch, self.train_losses, self.val_losses = load_checkpoint_dynamic(
            model=self.model,
            directory=self.config['checkpoint_dir'],
            optimizer=self.optimizer,
            for_training=True
        )
        self.model.to(self.device)
        print(f"آموزش از epoch {self.start_epoch} ادامه می‌یابد.")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0
        total_intersection = torch.zeros(self.num_classes, device=self.device)
        total_union = torch.zeros(self.num_classes, device=self.device)

        current_criterion = self.criterion_ce
        if epoch >= self.finetune_epoch:
            current_criterion = self.criterion_lovasz

        train_loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]")
        
        for step, batch in enumerate(train_loop):
            batch = batch.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                outputs, labels = self.model(batch)
                loss = current_criterion(outputs, labels) 
                loss = loss / self.accumulation_steps 

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            
            preds = torch.argmax(outputs, dim=1)
            _, _, batch_intersection, batch_union = calculate_metrics(outputs, labels, self.num_classes)
            total_intersection += batch_intersection
            total_union += batch_union
            total_correct += (preds == labels).sum().item()
            total_points += labels.numel()

            current_avg_loss = total_loss / (step + 1)
            train_loop.set_postfix(loss=f"{current_avg_loss:.4f}")

        avg_loss = total_loss / len(self.train_loader)
        oa = total_correct / total_points if total_points > 0 else 0.0
        iou_per_class = total_intersection / (total_union + 1e-8)
        valid_classes = torch.where(total_union > 0)[0]
        miou = torch.mean(iou_per_class[valid_classes]).item() if valid_classes.numel() > 0 else 0.0
        
        return avg_loss, oa, miou

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_points = 0
        total_intersection = torch.zeros(self.num_classes, device=self.device)
        total_union = torch.zeros(self.num_classes, device=self.device)

        validation_criterion = self.criterion_ce

        val_loop = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Val]")        
        with torch.no_grad():
            for batch in val_loop:
                batch = batch.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    outputs, labels = self.model(batch)
                    loss = validation_criterion(outputs, labels) 

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                _, _, batch_intersection, batch_union = calculate_metrics(outputs, labels, self.num_classes)
                total_intersection += batch_intersection
                total_union += batch_union
                total_correct += (preds == labels).sum().item()
                total_points += labels.numel()

        avg_loss = total_loss / len(self.val_loader)
        oa = total_correct / total_points if total_points > 0 else 0.0
        iou_per_class = total_intersection / (total_union + 1e-8)
        valid_classes = torch.where(total_union > 0)[0]
        miou = torch.mean(iou_per_class[valid_classes]).item() if valid_classes.numel() > 0 else 0.0
        
        return avg_loss, oa, miou

    def train(self):
        print(f"شروع آموزش از epoch {self.start_epoch}...")
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            if epoch == self.finetune_epoch:
                print("\n" + "="*50)
                print(f"STARTING FINE-TUNE PHASE (Epoch {epoch})")
                print(f"Switching to Lovasz-Softmax Loss and reducing LR to {self.finetune_lr}.")
                print("="*50 + "\n")
                for g in self.optimizer.param_groups:
                    g['lr'] = self.finetune_lr

            train_loss, train_oa, train_miou = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            val_loss, val_oa, val_miou = self._validate_epoch(epoch)
            self.val_losses.append(val_loss)

            if self.scheduler and epoch < self.finetune_epoch:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            print(
                f"\nEpoch {epoch+1} Summary:"
                f"\n  Train -> Loss: {train_loss:.4f}, OA: {train_oa:.4f}, mIoU: {train_miou:.4f}"
                f"\n  Val   -> Loss: {val_loss:.4f}, OA: {val_oa:.4f}, mIoU: {val_miou:.4f}"
                f"\n  LR: {current_lr:.6f}\n"
            )

            save_checkpoint(
                self.model, self.optimizer, epoch + 1, 
                self.train_losses, self.val_losses, 
                self.config['checkpoint_dir']
            )
        
        print("آموزش به پایان رسید.")