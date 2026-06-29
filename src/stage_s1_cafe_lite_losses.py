"""
Stage S1 CAFE-lite Same-Split Baseline Loss Functions.
Implements CrossEntropyLoss, Class-Balanced Loss, and Ambiguity Regularization.
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

def compute_cafe_lite_loss(
    logits, targets, ambiguity_score, similarity_score,
    loss_type="standard", class_weights=None, class_priors=None, tau=0.5,
    w_ambiguity_reg=0.1, eps=1e-12
):
    """
    Computes total loss for CAFE-lite model.
    
    Args:
        logits: Model predictions of shape [B, 6]
        targets: Labels of shape [B]
        ambiguity_score: Ambiguity gate values of shape [B]
        similarity_score: Cosine similarity values of shape [B]
        loss_type: "standard" (standard CE) or "balanced" (class-balanced/weighted CE)
        class_weights: Class weights tensor of shape [6] (used for balanced CE)
        class_priors: Class priors tensor of shape [6] (used for logit-adjusted CE if provided)
        tau: Logit adjustment temperature parameter
        w_ambiguity_reg: Weight of the ambiguity regularization term
        
    Returns:
        dict: containing 'loss_total', 'loss_ce', 'loss_amb_reg' and diagnostics
    """
    # 1. Sanity Checks
    for name, tensor in [("logits", logits), ("targets", targets), ("ambiguity_score", ambiguity_score)]:
        n_nan, n_inf = check_tensor_sanity(tensor, name)
        if n_nan > 0 or n_inf > 0:
            raise ValueError(f"[-] ERROR: {name} contains {n_nan} NaNs and {n_inf} Infs before loss computation.")
            
    # 2. Main Classification Loss
    if loss_type == "balanced":
        if class_priors is not None:
            # Logit-adjusted class-balanced cross entropy
            adjustments = tau * torch.log(class_priors + eps)
            adjusted_logits = logits + adjustments.unsqueeze(0)
            loss_ce = F.cross_entropy(adjusted_logits, targets, weight=class_weights)
        else:
            # Standard weighted cross entropy
            loss_ce = F.cross_entropy(logits, targets, weight=class_weights)
    else:
        # Standard unweighted cross entropy
        loss_ce = F.cross_entropy(logits, targets)
        
    # 3. Ambiguity Regularization
    # We want ambiguity score 'a' to align with the cross-modal disagreement (1 - normalized_similarity).
    # Similarity score is cosine similarity, which ranges from -1 to 1.
    # Normalize similarity to [0, 1]
    sim_normalized = (similarity_score + 1.0) / 2.0
    disagreement = 1.0 - sim_normalized
    
    # Regularization loss: MSE between ambiguity gate 'a' and disagreement
    loss_amb_reg = F.mse_loss(ambiguity_score, disagreement)
    
    # 4. Total Loss Combination
    total_loss = loss_ce + w_ambiguity_reg * loss_amb_reg
    
    # Final check on output losses
    n_nan_tot, n_inf_tot = check_tensor_sanity(total_loss, "total_loss")
    if n_nan_tot > 0 or n_inf_tot > 0:
        raise ValueError(f"[-] ERROR: Calculated loss is invalid: NaN={n_nan_tot}, Inf={n_inf_tot}")
        
    return {
        "loss_total": total_loss,
        "loss_ce": loss_ce,
        "loss_amb_reg": loss_amb_reg
    }
