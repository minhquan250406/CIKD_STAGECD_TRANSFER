"""
Stage I-F1: Feature-Refresh CIKD++-RT Model.
Defines StageIFFeatureRefreshCIKDPP which wraps a frozen F4 anchor
and updates representations via trainable adapters and a Residual Feature Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextFeatureAdapter(nn.Module):
    """
    Adapts text CLS feature from 768 to 256 dim.
    """
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.mlp(x)

class VisualPatchAdapter(nn.Module):
    """
    Performs attention-based pooling on image patches from 512 to 256 dim.
    """
    def __init__(self, in_dim=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.patch_proj = nn.Linear(in_dim, out_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.xavier_uniform_(self.query)
        self.Wq = nn.Linear(out_dim, out_dim)
        self.Wk = nn.Linear(out_dim, out_dim)
        self.Wv = nn.Linear(out_dim, out_dim)
        self.out_ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, 49, 512]
        patches = self.patch_proj(x)  # [B, 49, 256]
        B = patches.shape[0]
        
        q = self.Wq(self.query.expand(B, -1, -1))  # [B, 1, 256]
        k = self.Wk(patches)  # [B, 49, 256]
        v = self.Wv(patches)  # [B, 49, 256]
        
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)  # [B, 1, 49]
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, 1, 49]
        
        pooled = torch.matmul(attn_weights, v).squeeze(1)  # [B, 256]
        pooled = self.out_ln(pooled)
        return self.dropout(pooled)

class KGRelationAdapter(nn.Module):
    """
    Combines KG features and relation embeddings, adapting them to 256 dim.
    """
    def __init__(self, num_relations, kg_dim=100, relation_emb_dim=64, out_dim=256, dropout=0.1):
        super().__init__()
        self.relation_embed = nn.Embedding(num_relations, relation_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(kg_dim + relation_emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, kg_features, relation_ids):
        rel_emb = self.relation_embed(relation_ids)  # [B, relation_emb_dim]
        kg_rel = torch.cat([kg_features, rel_emb], dim=-1)  # [B, kg_dim + relation_emb_dim]
        return self.mlp(kg_rel)

class ResidualFeatureTransformer(nn.Module):
    """
    Fuses all token representations and computes final residual logit delta.
    """
    def __init__(self, d_model=256, num_layers=2, num_heads=4, dropout=0.1, num_classes=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, num_classes)
        
    def forward(self, seq):
        # seq: [B, N, d_model]
        encoded = self.transformer_encoder(seq)  # [B, N, d_model]
        pooled = encoded.mean(dim=1)  # [B, d_model]
        return self.out_proj(pooled)

class StageIFFeatureRefreshCIKDPP(nn.Module):
    """
    Stage I-F model: Frozen F4 anchor + refreshed adapters + Residual Feature Transformer.
    """
    def __init__(self, f4_model, num_relations, kg_dim=100, relation_emb_dim=64,
                 gamma=0.2, use_patch_adapter=True, use_kg_relation_adapter=True,
                 use_tvcs_zv=True, d_model=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.f4_model = f4_model
        
        # Freeze F4 Backbone
        for param in self.f4_model.parameters():
            param.requires_grad = False
            
        self.gamma = gamma
        self.use_patch_adapter = use_patch_adapter
        self.use_kg_relation_adapter = use_kg_relation_adapter
        self.use_tvcs_zv = use_tvcs_zv
        
        # 1. Text adapter
        self.text_adapter = TextFeatureAdapter(in_dim=768, hidden_dim=512, out_dim=d_model, dropout=dropout)
        
        # 2. Image global projection (from 512 to 256)
        self.image_global_proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 3. Visual patch adapter
        if self.use_patch_adapter:
            self.patch_adapter = VisualPatchAdapter(in_dim=512, out_dim=d_model, dropout=dropout)
        else:
            self.patch_adapter = None
            
        # 4. KG relation adapter
        if self.use_kg_relation_adapter:
            self.kg_relation_adapter = KGRelationAdapter(
                num_relations=num_relations,
                kg_dim=kg_dim,
                relation_emb_dim=relation_emb_dim,
                out_dim=d_model,
                dropout=dropout
            )
        else:
            self.kg_relation_adapter = None
            
        # 5. Visual evidence & tvcs projections (from F4)
        if self.use_tvcs_zv:
            self.z_v_proj = nn.Sequential(
                nn.Linear(512, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            self.tvcs_score_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
        else:
            self.z_v_proj = None
            self.tvcs_score_proj = None
            
        # 6. Transformer
        self.residual_transformer = ResidualFeatureTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_classes=6
        )

    def forward(self, text_features, image_global_features, image_patch_features, kg_features, relation_ids, baseline_logits, gamma_override=None):
        # 1. Run F4 Backbone (weights frozen, but run forward pass)
        f4_outputs = self.f4_model(
            text_features=text_features,
            image_global_features=image_global_features,
            image_patch_features=image_patch_features,
            kg_features=kg_features,
            relation_ids=relation_ids,
            baseline_logits=baseline_logits,
            ablation_no_c_emb=True  # As in F4 no_c_emb setup
        )
        
        f4_logits = f4_outputs['logits_final']
        z_v = f4_outputs['z_v']          # [B, 512]
        c_logit = f4_outputs['c_logit']  # [B]
        tvcs_score = torch.sigmoid(c_logit).unsqueeze(-1)  # [B, 1]
        
        # 2. Get refreshed features
        text_refresh = self.text_adapter(text_features)  # [B, 256]
        image_global_proj = self.image_global_proj(image_global_features)  # [B, 256]
        
        tokens = [
            text_refresh.unsqueeze(1),
            image_global_proj.unsqueeze(1)
        ]
        
        # Optional adapters
        if self.use_patch_adapter and self.patch_adapter is not None:
            patch_refresh = self.patch_adapter(image_patch_features)  # [B, 256]
            tokens.append(patch_refresh.unsqueeze(1))
        else:
            patch_refresh = None
            
        if self.use_kg_relation_adapter and self.kg_relation_adapter is not None:
            kg_refresh = self.kg_relation_adapter(kg_features, relation_ids)  # [B, 256]
            tokens.append(kg_refresh.unsqueeze(1))
        else:
            kg_refresh = None
            
        if self.use_tvcs_zv and self.z_v_proj is not None and self.tvcs_score_proj is not None:
            z_v_proj = self.z_v_proj(z_v)  # [B, 256]
            tvcs_score_proj = self.tvcs_score_proj(tvcs_score)  # [B, 256]
            tokens.append(z_v_proj.unsqueeze(1))
            tokens.append(tvcs_score_proj.unsqueeze(1))
            
        # 3. Fuse in ResidualFeatureTransformer
        seq = torch.cat(tokens, dim=1)  # [B, N, 256]
        delta_new = self.residual_transformer(seq)  # [B, 6]
        
        # 4. Final combination
        gamma = gamma_override if gamma_override is not None else self.gamma
        final_logits = f4_logits + gamma * delta_new
        
        return {
            "logits_final": final_logits,
            "delta_new": delta_new,
            "f4_logits": f4_logits,
            "text_refresh": text_refresh,
            "patch_refresh": patch_refresh,
            "kg_refresh": kg_refresh,
            "z_v": z_v,
            "tvcs_score": tvcs_score.squeeze(-1)
        }
