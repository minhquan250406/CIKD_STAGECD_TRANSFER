"""
Stage I-F1: Losses configuration and definitions.
Implements the 5-part Stage I-F loss function:
  L_total = 1.00 * BalancedCE_6way
          + 0.20 * FocalLoss_classes_1_2_5
          + 0.10 * CK_guard (protects class 2 from classes 0 and 3)
          + 0.10 * KL_to_F4
          + 0.03 * delta_norm
Includes sanity checks for NaN/Inf.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def check_tensor_sanity(tensor, name="tensor"):
    """
    Checks if a tensor contains NaN or Inf values.
    Returns:
        nan_count (int), inf_count (int)
    """
    if tensor is None:
        return 0, 0
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())
    return nan_count, inf_count

def logit_adjusted_ce_loss(logits, targets, class_priors, class_weights, tau=0.5, eps=1e-12):
    """
    Computes class-balanced logit-adjusted cross entropy loss.
    """
    assert torch.all(targets >= 0) and torch.all(targets < logits.shape[1]), "Targets out of bounds for 6-way CE"
    adjustments = tau * torch.log(class_priors + eps)
    adjusted_logits = logits + adjustments.unsqueeze(0)
    loss = F.cross_entropy(adjusted_logits, targets, weight=class_weights)
    return loss

def focal_bottleneck_loss(logits, targets, bottleneck_classes=[1, 2, 5], gamma=1.5):
    """
    Computes multi-class Focal Loss on samples belonging to bottleneck classes in the batch.
    Safe return 0.0 if no such samples in batch.
    """
    device = logits.device
    mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
    for c in bottleneck_classes:
        mask = mask | (targets == c)
        
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
        
    eligible_logits = logits[mask]
    eligible_targets = targets[mask]
    
    ce_loss = F.cross_entropy(eligible_logits, eligible_targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_term = (1.0 - pt) ** gamma
    loss = focal_term * ce_loss
    return loss.mean()

def ck_guard_loss(logits_final, targets):
    """
    CK Guard Loss: Protects CK (class 2) from confusion into Real (class 0) and text-based fake (class 3).
    Computes BCE loss on the conditional probability P(class 2 | class in {0, 2, 3}).
    Target mapping:
      class 0 or class 3 -> 0
      class 2 -> 1
    """
    mask = (targets == 0) | (targets == 2) | (targets == 3)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_final.device)
        
    eligible_logits = logits_final[mask]
    eligible_targets = targets[mask]
    
    # Logit of P(class 2 | class in {0, 2, 3})
    logit_class2 = eligible_logits[:, 2]
    # Sum exp for other classes (0 and 3)
    logit_others = torch.logsumexp(torch.stack([eligible_logits[:, 0], eligible_logits[:, 3]], dim=1), dim=1)
    binary_logits = logit_class2 - logit_others
    
    y_binary = (eligible_targets == 2).float()
    loss = F.binary_cross_entropy_with_logits(binary_logits, y_binary)
    return loss

def kl_to_f4_loss(f4_logits, final_logits, temperature=1.0):
    """
    Computes KL divergence between frozen F4 outputs and the final adapted predictions.
    """
    p = F.softmax(f4_logits / temperature, dim=-1)
    log_q = F.log_softmax(final_logits / temperature, dim=-1)
    loss = F.kl_div(log_q, p, reduction='batchmean') * (temperature ** 2)
    return loss

def delta_norm_loss(delta_new):
    """
    Computes L2 regularization on the transformer's delta_new output.
    """
    loss = torch.mean(delta_new ** 2)
    return loss

def compute_total_loss(
    logits_final, delta_new, f4_logits, targets,
    class_priors, class_weights, bottleneck_classes=[1, 2, 5],
    tau_logit_adjust=0.5, focal_gamma=1.5, kl_temperature=1.0,
    w_balanced=1.00, w_focal=0.20, w_ck_guard=0.10, w_kl=0.10, w_delta_norm=0.03
):
    """
    Aggregates all components of the total Stage I-F loss:
      L_total = w_balanced * L_balanced + w_focal * L_focal + w_ck_guard * L_ck_guard + w_kl * L_kl + w_delta_norm * L_delta_norm
    """
    loss_6way = logit_adjusted_ce_loss(logits_final, targets, class_priors, class_weights, tau=tau_logit_adjust)
    loss_focal = focal_bottleneck_loss(logits_final, targets, bottleneck_classes, gamma=focal_gamma)
    loss_ck_guard = ck_guard_loss(logits_final, targets)
    loss_kl = kl_to_f4_loss(f4_logits, logits_final, temperature=kl_temperature)
    loss_res = delta_norm_loss(delta_new)
    
    total_loss = (
        w_balanced * loss_6way +
        w_focal * loss_focal +
        w_ck_guard * loss_ck_guard +
        w_kl * loss_kl +
        w_delta_norm * loss_res
    )
    
    return {
        "loss_total": total_loss,
        "loss_6way": loss_6way,
        "loss_focal": loss_focal,
        "loss_ck_guard": loss_ck_guard,
        "loss_kl": loss_kl,
        "loss_residual": loss_res
    }
