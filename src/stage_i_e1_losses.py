"""
Stage I-E1: Losses configuration and definitions.
Implements the 5-part adapter loss function:
  L_total = 1.00 * L_balanced_6way
          + 0.30 * L_focal_bottleneck_classes
          + 0.10 * L_CK_guard
          + 0.08 * KL(F4 || final)
          + 0.02 * ||adapter_delta||^2
Includes safe checks for NaN/Inf propagation.
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
    L = -log( exp(z_y + tau*log(pi_y)) / sum_j exp(z_j + tau*log(pi_j)) )
    """
    # Verify bounds
    assert torch.all(targets >= 0) and torch.all(targets < logits.shape[1]), "Targets out of bounds for 6-way CE"
    
    # Calculate offset
    adjustments = tau * torch.log(class_priors + eps)  # [C]
    adjusted_logits = logits + adjustments.unsqueeze(0)  # [B, C]
    
    loss = F.cross_entropy(adjusted_logits, targets, weight=class_weights)
    return loss

def focal_bottleneck_loss(logits, targets, bottleneck_classes, gamma=1.5):
    """
    Computes multi-class Focal Loss on samples belonging to bottleneck classes in the batch.
    L = (1 - pt)^gamma * CE(logits, targets)
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
    CK Guard Loss: Protects CK (class 2) from being confused with Real (class 0, 1).
    Computes BCE loss on the conditional probability P(class 2 | class in {0, 1, 2}).
    Target mapping:
      Real: {0, 1} -> 0
      CK-Fake: 2 -> 1
    """
    mask = (targets == 0) | (targets == 1) | (targets == 2)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_final.device)
        
    eligible_logits = logits_final[mask]
    eligible_targets = targets[mask]
    
    # Logit of P(class 2 | class in {0, 1, 2})
    logit_class2 = eligible_logits[:, 2]
    logit_real = torch.logsumexp(eligible_logits[:, :2], dim=-1)
    binary_logits = logit_class2 - logit_real
    
    y_binary = (eligible_targets == 2).float()
    loss = F.binary_cross_entropy_with_logits(binary_logits, y_binary)
    return loss

def kl_f4_final_loss(f4_logits, final_logits, temperature=1.0):
    """
    Computes KL divergence between frozen F4 outputs and the final adapted predictions.
    """
    p = F.softmax(f4_logits / temperature, dim=-1)
    log_q = F.log_softmax(final_logits / temperature, dim=-1)
    # F.kl_div expects (input_log_probs, target_probabilities)
    loss = F.kl_div(log_q, p, reduction='batchmean') * (temperature ** 2)
    return loss

def adapter_delta_l2_loss(adapter_delta):
    """
    Computes L2 regularization on adapter output delta.
    """
    loss = torch.mean(adapter_delta ** 2)
    return loss

def compute_adapter_total_loss(
    logits_final, logits_delta, f4_logits, targets,
    class_priors, class_weights, bottleneck_classes,
    tau_logit_adjust, focal_gamma=1.5, kl_temperature=1.0,
    w_balanced=1.00, w_focal=0.30, w_ck_guard=0.10, w_kl=0.08, w_reg=0.02
):
    """
    Aggregates all components of the total adapter loss:
      L_total = w_balanced * L_6way
              + w_focal * L_focal
              + w_ck_guard * L_ck_guard
              + w_kl * L_kl
              + w_reg * L_residual
    """
    loss_6way = logit_adjusted_ce_loss(logits_final, targets, class_priors, class_weights, tau=tau_logit_adjust)
    loss_focal = focal_bottleneck_loss(logits_final, targets, bottleneck_classes, gamma=focal_gamma)
    loss_ck_guard = ck_guard_loss(logits_final, targets)
    loss_kl = kl_f4_final_loss(f4_logits, logits_final, temperature=kl_temperature)
    loss_res = adapter_delta_l2_loss(logits_delta)
    
    total_loss = (
        w_balanced * loss_6way +
        w_focal * loss_focal +
        w_ck_guard * loss_ck_guard +
        w_kl * loss_kl +
        w_reg * loss_res
    )
    
    return {
        "loss_total": total_loss,
        "loss_6way": loss_6way,
        "loss_focal": loss_focal,
        "loss_ck_guard": loss_ck_guard,
        "loss_kl": loss_kl,
        "loss_residual": loss_res
    }
