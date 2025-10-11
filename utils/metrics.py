import torch

def calculate_metrics(outputs, labels, num_classes):
    """
    محاسبه متریک‌های Overall Accuracy و Intersection-over-Union برای یک بچ.

    Args:
        outputs (torch.Tensor): خروجی‌های خام مدل (logits) با ابعاد [N, num_classes].
        labels (torch.Tensor): برچسب‌های واقعی با ابعاد [N].
        num_classes (int): تعداد کل کلاس‌ها.

    Returns:
        tuple: شامل:
            - overall_accuracy (float): دقت کلی بچ.
            - per_class_iou (torch.Tensor): تانسوری از IoU برای هر کلاس.
    """
    # گرفتن پیش‌بینی‌ها با انتخاب کلاسی که بیشترین امتیاز را دارد
    preds = torch.argmax(outputs, dim=1)
    
    # --- محاسبه Overall Accuracy (OA) ---
    correct_predictions = (preds == labels).sum().item()
    total_points = labels.numel()
    overall_accuracy = correct_predictions / total_points if total_points > 0 else 0.0

    # --- محاسبه Intersection over Union (IoU) برای هر کلاس ---
    # برای جلوگیری از خطا در بچ‌هایی که همه کلاس‌ها را ندارند، از `bincount` استفاده می‌کنیم
    intersection = torch.zeros(num_classes, device=outputs.device)
    union = torch.zeros(num_classes, device=outputs.device)
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        
        intersection[cls] = (pred_mask & label_mask).sum()
        union[cls] = (pred_mask | label_mask).sum()

    # IoU برای هر کلاس محاسبه می‌شود. برای جلوگیری از تقسیم بر صفر، یک اپسیلون کوچک اضافه می‌شود.
    per_class_iou = intersection / (union + 1e-8)

    return overall_accuracy, per_class_iou, intersection, union