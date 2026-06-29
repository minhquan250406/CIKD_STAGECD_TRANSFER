"""
Stage I-BC: Losses configuration and definitions.
Implements logit-adjusted cross-entropy, auxiliary losses (binary, CK-vs-real),
KL anchor regularization, and residual L2 constraints.
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
    adjustments = tau * torch.log(class_priors + eps) # [C]
    adjusted_logits = logits + adjustments.unsqueeze(0) # [B, C]
    
    loss = F.cross_entropy(adjusted_logits, targets, weight=class_weights)
    return loss

def binary_fake_real_loss(binary_logits, targets):
    """
    Computes binary fake/real classification loss (BCE with logits).
    Target mapping:
      Real: {0, 1} -> 0
      Fake: {2, 3, 4, 5} -> 1
    """
    binary_targets = (targets >= 2).float() # [B]
    loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
    return loss

def masked_ck_real_loss(ck_real_logits, targets):
    """
    Computes CK-vs-real classification loss (BCE with logits).
    Masked only on Real (0, 1) and CK-Fake (2) samples.
    Target mapping on active mask:
      Real: {0, 1} -> 0
      CK-Fake: 2 -> 1
    """
    mask = (targets == 0) | (targets == 1) | (targets == 2)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=ck_real_logits.device)
    
    eligible_logits = ck_real_logits[mask]
    eligible_targets = (targets[mask] == 2).float()
    
    loss = F.binary_cross_entropy_with_logits(eligible_logits, eligible_targets)
    return loss

def kl_anchor_loss(logits_final, logits_base, temperature=1.0):
    """
    Computes KL divergence between final predictions and frozen baseline anchor.
    """
    p = F.softmax(logits_base / temperature, dim=-1)
    log_q = F.log_softmax(logits_final / temperature, dim=-1)
    # F.kl_div expects (input_log_probs, target_probabilities)
    loss = F.kl_div(log_q, p, reduction='batchmean') * (temperature ** 2)
    return loss

def residual_delta_l2_loss(logits_delta):
    """
    Computes L2 regularization on residual logits delta.
    """
    loss = torch.mean(logits_delta ** 2)
    return loss

def compute_multitask_total_loss(
    logits_final, logits_delta, binary_logits, ck_real_logits, targets, logits_base,
    class_priors, class_weights, tau_logit_adjust,
    binary_loss_weight=0.25, ck_real_loss_weight=0.35,
    kl_anchor_weight=0.05, residual_reg_weight=0.01,
    kl_temperature=1.0
):
    """
    Aggregates all components of the total loss:
      L_total = 1.00 * L_6way_balanced
              + binary_loss_weight * L_binary
              + ck_real_loss_weight * L_CK_real
              + kl_anchor_weight * L_KL_anchor
              + residual_reg_weight * L_residual_reg
    """
    # 6-way Class-Balanced Logit-Adjusted CE
    loss_6way = logit_adjusted_ce_loss(logits_final, targets, class_priors, class_weights, tau=tau_logit_adjust)
    
    # Binary CE
    loss_binary = binary_fake_real_loss(binary_logits, targets)
    
    # Masked CK-vs-Real CE
    loss_ck_real = masked_ck_real_loss(ck_real_logits, targets)
    
    # KL Anchor
    loss_kl = kl_anchor_loss(logits_final, logits_base, temperature=kl_temperature)
    
    # Residual L2 Regularization
    loss_res = residual_delta_l2_loss(logits_delta)
    
    # Total loss combination
    total_loss = (
        1.00 * loss_6way +
        binary_loss_weight * loss_binary +
        ck_real_loss_weight * loss_ck_real +
        kl_anchor_weight * loss_kl +
        residual_reg_weight * loss_res
    )
    
    return {
        "loss_total": total_loss,
        "loss_6way": loss_6way,
        "loss_binary": loss_binary,
        "loss_ck_real": loss_ck_real,
        "loss_kl": loss_kl,
        "loss_residual": loss_res
    }
