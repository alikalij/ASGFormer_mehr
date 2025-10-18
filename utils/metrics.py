# utils/metrics.py
import torch

def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    محاسبه متریک‌های Overall Accuracy و Intersection-over-Union برای یک بچ.
    این تابع intersection و union خام را برای agregat کردن برمی‌گرداند.

    Args:
        outputs (torch.Tensor): خروجی‌های خام مدل (logits) با ابعاد [N, num_classes].
        labels (torch.Tensor): برچسب‌های واقعی با ابعاد [N].
        num_classes (int): تعداد کل کلاس‌ها.

    Returns:
        tuple: شامل:
            - overall_accuracy (float): دقت کلی بچ (نسبت تعداد نقاط درست پیش‌بینی شده).
            - per_class_iou (torch.Tensor): تانسوری از IoU برای هر کلاس در این بچ [num_classes].
            - intersection (torch.Tensor): تانسور تعداد نقاط اشتراک برای هر کلاس [num_classes].
            - union (torch.Tensor): تانسور تعداد نقاط اجتماع برای هر کلاس [num_classes].
    """
    # گرفتن پیش‌بینی‌ها با انتخاب کلاسی که بیشترین امتیاز را دارد
    preds = torch.argmax(outputs, dim=1)
    
    # اطمینان از اینکه labels و preds در یک دستگاه هستند
    labels = labels.to(outputs.device)
    
    # --- محاسبه Overall Accuracy (OA) ---
    correct_predictions = (preds == labels).sum().item()
    total_points = labels.numel()
    overall_accuracy = correct_predictions / total_points if total_points > 0 else 0.0

    # --- محاسبه Intersection over Union (IoU) برای هر کلاس ---
    intersection = torch.zeros(num_classes, device=outputs.device, dtype=torch.long)
    union = torch.zeros(num_classes, device=outputs.device, dtype=torch.long)
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        
        intersection[cls] = (pred_mask & label_mask).sum()
        union[cls] = (pred_mask | label_mask).sum()

    # IoU برای هر کلاس محاسبه می‌شود. برای جلوگیری از تقسیم بر صفر، یک اپسیلون کوچک اضافه می‌شود.
    # توجه: این per_class_iou فقط برای بچ فعلی است و برای mIoU نهایی باید agregat شود.
    per_class_iou = intersection.float() / (union.float() + 1e-8)

    return overall_accuracy, per_class_iou, intersection, union