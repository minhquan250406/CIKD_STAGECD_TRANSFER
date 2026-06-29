"""
CIKD++ Residual TVCS-Transformer Model (CIKD++-RT).
Implements the TVCS Specialist, Residual Transformer Fusion, and the CIKDPPResidualTransformer wrapper.
"""

import numpy as np
import torch
import torch.nn as nn

class TVCSSpecialist(nn.Module):
    """
    TVCS Specialist Module.
    Responsible for KG-to-visual attention over image patch features and contradiction scoring.
    """
    def __init__(self, num_relations, kg_dim=100, relation_emb_dim=64, tvcs_dim=512, image_patch_dim=512, c_emb_dim=64):
        super().__init__()
        self.relation_embed = nn.Embedding(num_relations, relation_emb_dim)
        
        # Query projection MLP (combining KG features and relation embeddings)
        self.query_mlp = nn.Sequential(
            nn.Linear(kg_dim + relation_emb_dim, tvcs_dim),
            nn.ReLU(),
            nn.Linear(tvcs_dim, tvcs_dim)
        )
        
        # Image patch key/value projection
        self.patch_proj = nn.Linear(image_patch_dim, tvcs_dim)
        
        # Attention projection heads
        self.Wq = nn.Linear(tvcs_dim, tvcs_dim)
        self.Wk = nn.Linear(tvcs_dim, tvcs_dim)
        self.Wv = nn.Linear(tvcs_dim, tvcs_dim)
        
        # Contradiction prediction head (takes concat of [z_k_tvcs, z_v, diff, prod])
        self.c_logit_mlp = nn.Sequential(
            nn.Linear(tvcs_dim * 4, tvcs_dim),
            nn.ReLU(),
            nn.Linear(tvcs_dim, 1)
        )
        
        # Contradiction embedding head
        self.c_emb_mlp = nn.Sequential(
            nn.Linear(1, c_emb_dim),
            nn.ReLU(),
            nn.Linear(c_emb_dim, c_emb_dim)
        )

    def forward(self, kg_features, relation_ids, image_patch_features):
        """
        Args:
            kg_features: tensor of shape [B, kg_dim]
            relation_ids: tensor of shape [B]
            image_patch_features: tensor of shape [B, num_patches, image_patch_dim]
        Returns:
            z_v: Visual evidence representation [B, tvcs_dim]
            c_logit: Contradiction logit [B]
            c_emb: Contradiction embedding [B, c_emb_dim]
            attention_weights: Attention weights over patches [B, num_patches]
        """
        # Embed relation and concatenate with KG features
        rel_emb = self.relation_embed(relation_ids)  # [B, relation_emb_dim]
        kg_rel = torch.cat([kg_features, rel_emb], dim=-1)  # [B, kg_dim + relation_emb_dim]
        z_k_tvcs = self.query_mlp(kg_rel)  # [B, tvcs_dim]
        
        # Project patches
        patches_proj = self.patch_proj(image_patch_features)  # [B, num_patches, tvcs_dim]
        
        # KG-to-visual attention
        q = self.Wq(z_k_tvcs)  # [B, tvcs_dim]
        k = self.Wk(patches_proj)  # [B, num_patches, tvcs_dim]
        v = self.Wv(patches_proj)  # [B, num_patches, tvcs_dim]
        
        # Compute scaled dot-product attention
        attn_logits = torch.einsum('bd,bpd->bp', q, k) / (k.shape[-1] ** 0.5)  # [B, num_patches]
        attention_weights = torch.softmax(attn_logits, dim=-1)  # [B, num_patches]
        
        # Retrieve visual evidence
        z_v = torch.einsum('bp,bpd->bd', attention_weights, v)  # [B, tvcs_dim]
        
        # Contradiction scoring
        diff = torch.abs(z_k_tvcs - z_v)
        prod = z_k_tvcs * z_v
        c_input = torch.cat([z_k_tvcs, z_v, diff, prod], dim=-1)  # [B, tvcs_dim * 4]
        c_logit = self.c_logit_mlp(c_input).squeeze(-1)  # [B]
        
        # Contradiction embedding
        c_emb = self.c_emb_mlp(c_logit.unsqueeze(-1))  # [B, c_emb_dim]
        
        return z_v, c_logit, c_emb, attention_weights


class ResidualTransformerFusion(nn.Module):
    """
    Residual Transformer Fusion Module.
    Projects and fuses modalities and features using a Transformer Encoder, producing residual logits.
    """
    def __init__(self, text_dim=768, image_global_dim=512, kg_dim=100, relation_emb_dim=64, 
                 tvcs_dim=512, c_emb_dim=64, num_classes=6, d_model=256, num_layers=2, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        # Projection heads to project each token type to d_model
        self.proj_text = nn.Linear(text_dim, d_model)
        self.proj_img = nn.Linear(image_global_dim, d_model)
        self.proj_kg = nn.Linear(kg_dim, d_model)
        self.proj_rel = nn.Linear(relation_emb_dim, d_model)
        self.proj_zv = nn.Linear(tvcs_dim, d_model)
        self.proj_c_emb = nn.Linear(c_emb_dim, d_model)
        self.proj_logits = nn.Linear(num_classes, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output prediction projection
        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, text_features, image_global_features, kg_features, relation_embedding, 
                tvcs_visual_evidence, c_emb, baseline_logits):
        """
        Args:
            text_features: [B, text_dim]
            image_global_features: [B, image_global_dim]
            kg_features: [B, kg_dim]
            relation_embedding: [B, relation_emb_dim]
            tvcs_visual_evidence: [B, tvcs_dim]
            c_emb: [B, c_emb_dim]
            baseline_logits: [B, num_classes]
        Returns:
            logits_delta: [B, num_classes]
        """
        # Project inputs to d_model and unsqueeze to add sequence dimension [B, 1, d_model]
        t_tok = self.proj_text(text_features).unsqueeze(1)
        img_tok = self.proj_img(image_global_features).unsqueeze(1)
        kg_tok = self.proj_kg(kg_features).unsqueeze(1)
        rel_tok = self.proj_rel(relation_embedding).unsqueeze(1)
        zv_tok = self.proj_zv(tvcs_visual_evidence).unsqueeze(1)
        c_tok = self.proj_c_emb(c_emb).unsqueeze(1)
        logits_tok = self.proj_logits(baseline_logits).unsqueeze(1)
        
        # Sequence of tokens: [B, 7, d_model]
        seq = torch.cat([t_tok, img_tok, kg_tok, rel_tok, zv_tok, c_tok, logits_tok], dim=1)
        
        # Run Transformer Encoder
        encoded = self.transformer_encoder(seq)  # [B, 7, d_model]
        
        # Mean pooling across token representations
        pooled = encoded.mean(dim=1)  # [B, d_model]
        
        # Compute delta logits
        logits_delta = self.out_proj(pooled)  # [B, num_classes]
        
        return logits_delta


class CIKDPPResidualTransformer(nn.Module):
    """
    CIKD++ Residual Transformer Wrapper.
    Orchestrates TVCS Specialist and ResidualTransformerFusion.
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
        
        # Initialize sub-modules
        self.tvcs_specialist = TVCSSpecialist(
            num_relations=num_relations,
            kg_dim=kg_dim,
            relation_emb_dim=relation_emb_dim,
            tvcs_dim=tvcs_dim,
            image_patch_dim=image_patch_dim,
            c_emb_dim=c_emb_dim
        )
        
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

    def forward(self, text_features, image_global_features, image_patch_features, kg_features, relation_ids, baseline_logits,
                ablation_no_c_emb=False, ablation_no_residual=False, ablation_global_only=False):
        """
        Args:
            text_features: [B, text_dim]
            image_global_features: [B, image_global_dim]
            image_patch_features: [B, num_patches, image_patch_dim]
            kg_features: [B, kg_dim]
            relation_ids: [B]
            baseline_logits: [B, num_classes]
            ablation_no_c_emb: Zeroes out contradiction embedding if True
            ablation_no_residual: Sets residual scaling alpha to 0 if True
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
        
        # Apply global_only ablation: Zero out visual evidence z_v
        if ablation_global_only:
            z_v = torch.zeros_like(z_v)
            
        # Apply no_c_emb ablation: Zero out contradiction embedding c_emb
        if ablation_no_c_emb:
            c_emb = torch.zeros_like(c_emb)
            
        # Retrieve relation embedding
        relation_embedding = self.tvcs_specialist.relation_embed(relation_ids)
        
        # Call Residual Transformer Fusion
        logits_delta = self.residual_transformer(
            text_features=text_features,
            image_global_features=image_global_features,
            kg_features=kg_features,
            relation_embedding=relation_embedding,
            tvcs_visual_evidence=z_v,
            c_emb=c_emb,
            baseline_logits=baseline_logits
        )
        
        # Compute sigmoid-scaled and alpha_max capped alpha parameter
        if ablation_no_residual:
            alpha = torch.zeros(1, device=baseline_logits.device)
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
            "attention": attention
        }
