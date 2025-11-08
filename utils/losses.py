# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
کد Lovász-Softmax بر اساس پیاده‌سازی رسمی نویسندگان مقاله Lovász-Softmax.
https://github.com/berman-maxim/LovaszSoftmax
"""

def lovasz_grad(gt_sorted):
    """
    محاسبه گرادیان Lovász
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - torch.cumsum(gt_sorted, dim=0)
    union = gts + torch.cumsum(1.0 - gt_sorted, dim=0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    نسخه Flat از Lovasz-Softmax
    probas: [P, C] متغیر (خروجی softmax)
    labels: [P] هدف (ground truth)
    classes: 'all' or 'present'
    """
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    
    for c in class_to_sum:
        fg = (labels == c).float() # [P]
        if classes == 'present' and fg.sum() == 0:
            continue
        
        probas_c = probas[:, c] # [P]
        errors = (fg - probas_c).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
        
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, classes='present', ignore_index=None):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        logits: [N, C, *] (خروجی خام مدل)
        labels: [N, *] (برچسب‌های واقعی)
        """
        # --- نادیده گرفتن ignore_index ---
        if self.ignore_index is not None:
            mask = (labels != self.ignore_index)
            logits = logits[mask]
            labels = labels[mask]
            if logits.numel() == 0:
                return 0.0 # اگر همه چیز ignore شد

        # --- اعمال Softmax ---
        # ورودی باید Probabilities باشد، نه Logits
        probas = F.softmax(logits, dim=1)
        
        # --- Flat کردن ابعاد ---
        # [N, C, H, W] -> [N*H*W, C] یا [N, C] -> [N, C]
        if probas.dim() > 2:
            probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
            labels = labels.view(-1)
        # برای ابر نقاط که از قبل [N, C] است، نیازی به تغییر نیست

        return lovasz_softmax_flat(probas, labels, self.classes)