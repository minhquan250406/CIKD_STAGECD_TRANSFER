"""
Stage S1 CAFE-lite Same-Split Baseline Model.
Implements the CafeLiteSameSplitModel with cross-modal ambiguity estimation,
common/specific gating, and optional patch attention pooling.
"""

import math
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """
    Self-contained attention pooling layer.
    Pools patch tokens [B, N, D] into a single representation [B, D]
    using a learnable query vector.
    """
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        
    def forward(self, x):
        # x: [B, N, d_model]
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, d_model]
        k = self.key_proj(x)  # [B, N, d_model]
        v = self.value_proj(x)  # [B, N, d_model]
        
        # Calculate dot-product attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, 1, N]
        attn = torch.softmax(scores, dim=-1)  # [B, 1, N]
        
        pooled = torch.bmm(attn, v).squeeze(1)  # [B, d_model]
        return pooled

class CafeLiteSameSplitModel(nn.Module):
    """
    CAFE-style lightweight baseline model.
    No KG, relation IDs, TVCS, baseline residual anchor, or c_emb are used.
    
    1. Text projection: [768] -> d_model
    2. Image global projection: [512] -> d_model
    3. Optional patch attention pooling: [49, 512] -> d_model
    4. Cross-modal ambiguity estimation: similarity, ambiguity gate a in [0,1], common and specific representations
    5. Fusion: [text_proj, image_proj, ambiguity_feature, common_repr, specific_repr]
    6. Classifier: fused -> 6 logits
    """
    def __init__(self, d_model=256, use_patch_pooling=False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.use_patch_pooling = use_patch_pooling
        
        # 1. Projections
        self.proj_text = nn.Linear(768, d_model)
        self.proj_image_global = nn.Linear(512, d_model)
        
        if self.use_patch_pooling:
            self.proj_patch = nn.Linear(512, d_model)
            self.patch_pooling_layer = AttentionPooling(d_model)
            self.patch_mix_layer = nn.Linear(2 * d_model, d_model)
            
        # 2. Ambiguity & Similarity MLP
        # ambiguity_feature is concatenation of absolute difference [d_model] and product [d_model] -> size 2 * d_model
        self.ambiguity_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # 3. Common & Modality-specific representation layers
        self.common_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.text_specific_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.image_specific_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 4. Fusion and Classification
        # fused = [text_proj, image_proj, ambiguity_feature, common_repr, specific_repr]
        # sizes: d_model + d_model + 2*d_model + d_model + 2*d_model = 7 * d_model
        fused_dim = 7 * d_model
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 6)
        )
        
    def forward(self, text_features, image_features_global, image_features_patch=None):
        # 1. Project inputs
        text_proj = self.proj_text(text_features)  # [B, d_model]
        image_global_proj = self.proj_image_global(image_features_global)  # [B, d_model]
        
        # Process visual features (optional patch pooling)
        if self.use_patch_pooling:
            if image_features_patch is None:
                raise ValueError("image_features_patch must be provided when use_patch_pooling is True.")
            patches_proj = self.proj_patch(image_features_patch)  # [B, 49, d_model]
            pooled_patch = self.patch_pooling_layer(patches_proj)  # [B, d_model]
            # Mix global and local patch features
            image_proj_for_sim = self.patch_mix_layer(torch.cat([image_global_proj, pooled_patch], dim=-1))  # [B, d_model]
        else:
            image_proj_for_sim = image_global_proj
            
        # 2. Similarity and ambiguity estimation
        norm_text = torch.nn.functional.normalize(text_proj, p=2, dim=-1)
        norm_image = torch.nn.functional.normalize(image_proj_for_sim, p=2, dim=-1)
        similarity_score = torch.sum(norm_text * norm_image, dim=-1)  # [B]
        
        # Compute difference and product (disagreement/similarity features)
        diff_feat = torch.abs(text_proj - image_proj_for_sim)  # [B, d_model]
        prod_feat = text_proj * image_proj_for_sim  # [B, d_model]
        ambiguity_feature = torch.cat([diff_feat, prod_feat], dim=-1)  # [B, 2 * d_model]
        
        # Ambiguity gate 'a' in [0, 1]
        ambiguity_score = torch.sigmoid(self.ambiguity_mlp(ambiguity_feature)).squeeze(-1)  # [B]
        
        # 3. Derive Common & Specific representations
        # Common representation: active when ambiguity is low (1 - a)
        common_base = self.common_layer(torch.cat([text_proj, image_proj_for_sim], dim=-1))  # [B, d_model]
        common_repr = (1.0 - ambiguity_score.unsqueeze(-1)) * common_base  # [B, d_model]
        
        # Specific representation: active when ambiguity is high (a)
        text_spec = self.text_specific_layer(text_proj)  # [B, d_model]
        image_spec = self.image_specific_layer(image_proj_for_sim)  # [B, d_model]
        specific_base = torch.cat([text_spec, image_spec], dim=-1)  # [B, 2 * d_model]
        specific_repr = ambiguity_score.unsqueeze(-1) * specific_base  # [B, 2 * d_model]
        
        # 4. Fusion
        fused = torch.cat([
            text_proj,
            image_global_proj,  # keep the original global image projection as requested
            ambiguity_feature,
            common_repr,
            specific_repr
        ], dim=-1)  # [B, 7 * d_model]
        
        # 5. Classifier
        logits = self.classifier(fused)  # [B, 6]
        
        # 6. Diagnostics
        diagnostics = {
            "fused": fused,
            "text_proj_norm": torch.norm(text_proj, p=2, dim=-1).mean().item(),
            "image_proj_norm": torch.norm(image_proj_for_sim, p=2, dim=-1).mean().item(),
            "common_repr_norm": torch.norm(common_repr, p=2, dim=-1).mean().item(),
            "specific_repr_norm": torch.norm(specific_repr, p=2, dim=-1).mean().item(),
            "mean_ambiguity": ambiguity_score.mean().item(),
            "mean_similarity": similarity_score.mean().item()
        }
        
        return logits, ambiguity_score, similarity_score, diagnostics
