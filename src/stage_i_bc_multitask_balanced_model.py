"""
Stage I-BC: Multitask CIKD++ Model with class-balanced / logit-adjusted objectives and guardrail heads.
Extends CIKD++-RT no_c_emb architecture with auxiliary multitask heads.
"""

import numpy as np
import torch
import torch.nn as nn
from models.cikd_pp_rt import TVCSSpecialist, ResidualTransformerFusion

class StageIBCMultitaskCIKDPP(nn.Module):
    """
    Stage I-BC Multitask CIKD++ Model.
    Keeps the F4 no_c_emb residual TVCS-Transformer structure:
        logits_final = logits_base + alpha * delta_logits
    And adds auxiliary heads:
        1. fine6_head: handled directly by logits_final
        2. binary_head: fake/real auxiliary classification
        3. ck_real_head: CK-vs-real auxiliary classification
    """
    def __init__(self, num_relations, text_dim=768, image_global_dim=512, image_patch_dim=512,
                 num_patches=49, kg_dim=100, relation_emb_dim=64, c_emb_dim=64,
                 d_model=256, num_layers=2, num_heads=4, dropout=0.2,
                 alpha_init=0.2, alpha_max=0.5, tvcs_dim=512, num_classes=6):
        super().__init__()
        self.alpha_max = alpha_max
        
        # Map alpha_init back to raw logit space for sigmoid parameterization
        ratio = alpha_init / alpha_max
        ratio = max(min(ratio, 0.999), 0.001)  # safe clamp
        alpha_raw_val = np.log(ratio / (1.0 - ratio))
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_raw_val, dtype=torch.float32))
        
        # Initialize TVCS Specialist
        self.tvcs_specialist = TVCSSpecialist(
            num_relations=num_relations,
            kg_dim=kg_dim,
            relation_emb_dim=relation_emb_dim,
            tvcs_dim=tvcs_dim,
            image_patch_dim=image_patch_dim,
            c_emb_dim=c_emb_dim
        )
        
        # Initialize Residual Transformer Fusion
        self.residual_transformer = ResidualTransformerFusion(
            text_dim=text_dim,
            image_global_dim=image_global_dim,
            kg_dim=kg_dim,
            relation_emb_dim=relation_emb_dim,
            tvcs_dim=tvcs_dim,
            c_emb_dim=c_emb_dim,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Auxiliary heads from pooled transformer encoder representations
        self.binary_head = nn.Linear(d_model, 1)
        self.ck_real_head = nn.Linear(d_model, 1)

    def forward(self, text_features, image_global_features, image_patch_features, kg_features, relation_ids, baseline_logits,
                scheduled_alpha=None, ablation_no_c_emb=True, ablation_no_residual=False, ablation_global_only=False):
        """
        Args:
            text_features: [B, text_dim]
            image_global_features: [B, image_global_dim]
            image_patch_features: [B, num_patches, image_patch_dim]
            kg_features: [B, kg_dim]
            relation_ids: [B]
            baseline_logits: [B, num_classes]
            scheduled_alpha: Optional scheduled alpha to override learning parameter
            ablation_no_c_emb: Zeroes out contradiction embedding if True (True by default for no_c_emb)
            ablation_no_residual: Sets alpha to 0 if True
            ablation_global_only: Zeroes out TVCS visual evidence z_v if True
        Returns:
            dictionary of output tensors
        """
        # Call TVCS Specialist
        z_v, c_logit, c_emb, attention = self.tvcs_specialist(
            kg_features=kg_features,
            relation_ids=relation_ids,
            image_patch_features=image_patch_features
        )
        
        # Apply global_only ablation
        if ablation_global_only:
            z_v = torch.zeros_like(z_v)
            
        # Apply no_c_emb ablation
        if ablation_no_c_emb:
            c_emb = torch.zeros_like(c_emb)
            
        # Retrieve relation embedding
        relation_embedding = self.tvcs_specialist.relation_embed(relation_ids)
        
        # Extrapolate pooled token representations from Residual Transformer
        rt = self.residual_transformer
        t_tok = rt.proj_text(text_features).unsqueeze(1)
        img_tok = rt.proj_img(image_global_features).unsqueeze(1)
        kg_tok = rt.proj_kg(kg_features).unsqueeze(1)
        rel_tok = rt.proj_rel(relation_embedding).unsqueeze(1)
        zv_tok = rt.proj_zv(z_v).unsqueeze(1)
        c_tok = rt.proj_c_emb(c_emb).unsqueeze(1)
        logits_tok = rt.proj_logits(baseline_logits).unsqueeze(1)
        
        seq = torch.cat([t_tok, img_tok, kg_tok, rel_tok, zv_tok, c_tok, logits_tok], dim=1)
        encoded = rt.transformer_encoder(seq)
        pooled = encoded.mean(dim=1)  # [B, d_model]
        
        # Main residual projection
        logits_delta = rt.out_proj(pooled)  # [B, num_classes]
        
        # Auxiliary head projections (squeezed to [B])
        binary_logits = self.binary_head(pooled).squeeze(-1)
        ck_real_logits = self.ck_real_head(pooled).squeeze(-1)
        
        # Alpha calculation
        if ablation_no_residual:
            alpha = torch.zeros(1, device=baseline_logits.device)
        elif scheduled_alpha is not None:
            if isinstance(scheduled_alpha, torch.Tensor):
                alpha = scheduled_alpha.to(baseline_logits.device)
            else:
                alpha = torch.tensor(scheduled_alpha, dtype=torch.float32, device=baseline_logits.device)
        else:
            alpha = torch.sigmoid(self.alpha_raw) * self.alpha_max
            
        # Combine base logits and scaled residual logits
        logits_final = baseline_logits + alpha * logits_delta
        
        return {
            "logits_final": logits_final,
            "logits_delta": logits_delta,
            "logits_base": baseline_logits,
            "alpha": alpha,
            "c_logit": c_logit,
            "c_emb": c_emb,
            "z_v": z_v,
            "attention": attention,
            "binary_logits": binary_logits,
            "ck_real_logits": ck_real_logits,
            "pooled": pooled
        }
