"""
Stage I-E1: Bottleneck Adapter Model.
Defines the StageIE1BottleneckAdapterModel wrapper, which holds a frozen F4 model
and applies a class-bottleneck residual adapter correction using F4 and baseline outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StageIE1BottleneckAdapterModel(nn.Module):
    """
    Stage I-E1 Class-Bottleneck Residual Adapter Model.
    Wraps a frozen CIKDPPResidualTransformer (F4) model and implements the residual correction:
        final_logits = f4_logits + beta * class_mask * adapter_delta
    Where:
        - f4_logits: frozen F4 output logits
        - adapter_delta: output of a small trainable MLP correction head
        - class_mask: vector of size [6] detailing correction scaling per class
        - beta: a small scalar multiplier (e.g., 0.1, 0.2, 0.3)
    """
    def __init__(self, f4_model, beta=0.1, class_mask=None, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.f4_model = f4_model
        
        # Freeze all F4 parameters
        for param in self.f4_model.parameters():
            param.requires_grad = False
            
        self.beta = beta
        
        # Class mask: shape [6]
        if class_mask is None:
            class_mask = [1.0] * 6
        self.register_buffer("class_mask", torch.tensor(class_mask, dtype=torch.float32))
        
        # Small MLP correction head
        # Input features:
        # - F4 logits: [B, 6]
        # - Baseline logits: [B, 6]
        # - F4 probabilities: [B, 6]
        # - F4 entropy: [B, 1]
        # - TVCS score: [B, 1]
        # - TVCS visual evidence z_v: [B, 512]
        # Total input dim: 6 + 6 + 6 + 1 + 1 + 512 = 532
        input_dim = 532
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6)
        )
        
    def forward(self, text_features, image_global_features, image_patch_features, kg_features, relation_ids, baseline_logits):
        # 1. Run F4 (weights are frozen, but we run forward without torch.no_grad so that the overall graph works,
        # since F4 has no requires_grad parameters, this is perfectly memory efficient)
        f4_outputs = self.f4_model(
            text_features=text_features,
            image_global_features=image_global_features,
            image_patch_features=image_patch_features,
            kg_features=kg_features,
            relation_ids=relation_ids,
            baseline_logits=baseline_logits,
            ablation_no_c_emb=True  # As in F4 baseline/ablation setup
        )
        
        f4_logits = f4_outputs['logits_final']
        z_v = f4_outputs['z_v']
        c_logit = f4_outputs['c_logit']
        
        # 2. Extract input features for the adapter
        f4_probs = torch.softmax(f4_logits, dim=-1)
        
        # Entropy
        entropy = -torch.sum(f4_probs * torch.log(f4_probs + 1e-12), dim=-1, keepdim=True)
        
        # TVCS contradiction probability score
        tvcs_score = torch.sigmoid(c_logit).unsqueeze(-1)  # Shape [B, 1]
        
        # Concatenate features
        h_adapter = torch.cat([
            f4_logits,
            baseline_logits,
            f4_probs,
            entropy,
            tvcs_score,
            z_v
        ], dim=-1)  # Shape [B, 532]
        
        # 3. Compute adapter delta logits correction
        adapter_delta = self.mlp(h_adapter)  # Shape [B, 6]
        
        # 4. Final logit correction combination
        # class_mask is registered as buffer of shape [6]
        final_logits = f4_logits + self.beta * self.class_mask.unsqueeze(0) * adapter_delta
        
        return {
            "logits_final": final_logits,
            "logits_delta": adapter_delta,
            "f4_logits": f4_logits,
            "z_v": z_v,
            "c_logit": c_logit,
            "tvcs_score": tvcs_score.squeeze(-1)
        }
