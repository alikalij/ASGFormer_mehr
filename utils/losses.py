# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6, ignore_index=None):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)  
        n_classes = probs.shape[1]

        if logits.dim() == 4:
            B,C,H,W = logits.shape
            probs = probs.view(B, C, -1)
            target = target.view(B, -1)
        else:
            B,C,N = probs.shape
        target_onehot = F.one_hot(target.long(), num_classes=n_classes)  
        target_onehot = target_onehot.permute(0,2,1).float()  

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).unsqueeze(1)  
            probs = probs * mask
            target_onehot = target_onehot * mask

        numerator = 2 * (probs * target_onehot).sum(dim=2)  
        denominator = probs.sum(dim=2) + target_onehot.sum(dim=2)  
        dice = (numerator + self.eps) / (denominator + self.eps)
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
    if probas.dim() == 3:
        C, B, N = probas.shape
        probas = probas.permute(1,0,2).reshape(-1, C).permute(1,0)  
        labels = labels.view(-1)
    elif probas.dim() == 2:
        pass
    probas = probas.contiguous()
    labels = labels.contiguous()
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[:, valid]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax_flat(probas, labels, classes='present'):
    C, P = probas.size()
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
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        B, C, N = probas.shape
        total_loss = 0.0
        count = 0
        for b in range(B):
            prob = probas[b]  
            lab = labels[b]   
            prob_f, lab_f = flatten_probas(prob, lab, self.ignore_index)
            if lab_f.numel() == 0:
                continue
            loss = lovasz_softmax_flat(prob_f, lab_f, classes='present')
            total_loss += loss
            count += 1
        if count == 0:
            return probas.new_tensor(0.)
        return total_loss / count


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
            cw = torch.tensor(class_weights, dtype=torch.float)
            self.ce = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index, label_smoothing=label_smoothing)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)

        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def focal_loss(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        loss = ((1 - p_t) ** self.focal_gamma) * ce_loss
        if self.focal_alpha is not None:
            pass
        if self.ignore_index is not None:
            loss = loss[target != self.ignore_index]
        return loss.mean()

    def forward(self, logits, target):
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
