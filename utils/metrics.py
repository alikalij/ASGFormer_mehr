# utils/metrics.py
import torch

def calculate_metrics(outputs, labels, num_classes):
    preds = torch.argmax(outputs, dim=1)

    correct_predictions = (preds == labels).sum().item()
    total_points = labels.numel()
    overall_accuracy = correct_predictions / total_points if total_points > 0 else 0.0

    intersection = torch.zeros(num_classes, device=outputs.device)
    union = torch.zeros(num_classes, device=outputs.device)

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)

        intersection[cls] = (pred_mask & label_mask).sum()
        union[cls] = (pred_mask | label_mask).sum()

    per_class_iou = intersection / (union + 1e-8)

    return overall_accuracy, per_class_iou, intersection, union


def calculate_final_metrics(all_preds, all_labels, num_classes):
    all_preds = all_preds.cpu().flatten()
    all_labels = all_labels.cpu().flatten()

    correct = (all_preds == all_labels).sum().item()
    total = all_labels.numel()
    overall_acc = correct / total if total > 0 else 0.0

    total_intersection = torch.zeros(num_classes)
    total_union = torch.zeros(num_classes)

    for cls in range(num_classes):
        pred_mask = (all_preds == cls)
        label_mask = (all_labels == cls)
        total_intersection[cls] = (pred_mask & label_mask).sum()
        total_union[cls] = (pred_mask | label_mask).sum()

    iou_per_class = total_intersection / (total_union + 1e-6)
    
    valid_classes_gt = torch.where(torch.bincount(all_labels, minlength=num_classes) > 0)[0]
    mIoU = torch.mean(iou_per_class[valid_classes_gt]).item() if valid_classes_gt.numel() > 0 else 0.0

    return overall_acc, mIoU, iou_per_class.numpy()