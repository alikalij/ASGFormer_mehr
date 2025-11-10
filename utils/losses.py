# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    DiceLoss اصلاح‌شده برای ورودی [N_total, C]
    """
    def __init__(self, eps=1e-6, ignore_index=None):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        n_classes = probs.shape[1]
        
        # [N_total, C]
        target_onehot = F.one_hot(target.long(), num_classes=n_classes).float()
        
        # [N_total, C]
        probs = probs

        if self.ignore_index is not None:
            # [N_total]
            mask = (target != self.ignore_index)
            # Apply mask to flattened tensors
            probs = probs[mask]
            target_onehot = target_onehot[mask]

        if probs.numel() == 0:
            return probs.new_tensor(0.)

        # Sum over the N_total dimension for each class
        # numerator and denominator will be [C]
        numerator = 2 * (probs * target_onehot).sum(dim=0)
        denominator = probs.sum(dim=0) + target_onehot.sum(dim=0)
        
        dice = (numerator + self.eps) / (denominator + self.eps)
        
        # Average the dice score across all present classes
        loss = 1 - dice.mean()
        return loss

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    if gts == 0:
        return torch.zeros_like(gt_sorted)
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def flatten_probas(probas, labels, ignore=None):
    """
    utility function for Lovasz-Softmax,
    expects [C, N] and [N]
    """
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[:, valid]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    expects [C, N] and [N]
    """
    C, P = probas.size()
    if P == 0:
        return probas.new_tensor(0.)
        
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        errors = (fg - probas[c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return probas.new_tensor(0.)
    return sum(losses) / len(losses)

class LovaszSoftmax(nn.Module):
    """
    Lovasz-Softmax اصلاح‌شده برای ورودی [N_total, C]
    """
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # 1. Get Probas: [N_total, C]
        probas = F.softmax(logits, dim=1)
        
        # 2. Permute for lovasz_softmax_flat: [C, N_total]
        probas_permuted = probas.permute(1, 0)
        
        # 3. Flatten (handles ignore_index)
        probas_flat, labels_flat = flatten_probas(probas_permuted, labels, self.ignore_index)
        
        # 4. Calculate loss
        return lovasz_softmax_flat(probas_flat, labels_flat, classes='present')


class CombinedLoss(nn.Module):
    def __init__(self,
                 class_weights=None,
                 ignore_index=None,
                 alpha=1.0, beta=1.0, gamma=0.5,
                 label_smoothing=0.0,
                 use_focal=False,
                 focal_gamma=2.0,
                 focal_alpha=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index

        if class_weights is not None:
            # Fix UserWarning: use .clone().detach()
            cw = class_weights.clone().detach().float()
        else:
            cw = None

        self.ce = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index, label_smoothing=label_smoothing)

        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def focal_loss(self, logits, target):
        # This implementation already works with [N, C] and [N]
        ce_loss = F.cross_entropy(logits, target, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        loss = ((1 - p_t) ** self.focal_gamma) * ce_loss
        
        if self.focal_alpha is not None:
             # Handle alpha weighting if needed (currently pass)
            pass

        # Apply ignore_index mask *after* calculation
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            loss = loss[mask]
            
        return loss.mean()

    def forward(self, logits, target):
        # logits: [N_total, C]
        # target: [N_total]
        loss = 0.0
        
        if self.alpha > 0:
            if self.use_focal:
                loss_ce = self.focal_loss(logits, target)
            else:
                loss_ce = self.ce(logits, target)
            loss = loss + self.alpha * loss_ce

        if self.beta > 0:
            loss_lov = self.lovasz(logits, target)
            loss = loss + self.beta * loss_lov

        if self.gamma > 0:
            loss_dice = self.dice(logits, target)
            loss = loss + self.gamma * loss_dice

        return loss