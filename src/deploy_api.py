import os
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

# --- Model Definition ---
class CIKDCKBoostMoE(nn.Module):
    """
    CIKD CK-Boosted Residual Mixture of Experts (MoE) Model.
    Copied from run_stage_cd.py for self-contained deployment.
    """
    def __init__(self, num_relations, kg_dim=100):
        super().__init__()
        self.base_expert = nn.Sequential(
            nn.Linear(768 + 512 + kg_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        self.relation_embed = nn.Embedding(num_relations, 32)
        self.z_k_tvcs_mlp = nn.Sequential(
            nn.Linear(kg_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.z_k_cls_mlp = nn.Sequential(
            nn.Linear(kg_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.patch_proj = nn.Linear(512, 512)
        self.Wq = nn.Linear(512, 512)
        self.Wk = nn.Linear(512, 512)
        self.Wv = nn.Linear(512, 512)
        self.c_logit_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.c_emb_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.tvcs_expert = nn.Sequential(
            nn.Linear(768 + 512 + 256 + 512 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(768 + 512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.ck_boost_mlp = nn.Sequential(
            nn.Linear(512 + 512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, text_feats, img_global, img_patch, kg_feats, relation_ids):
        base_input = torch.cat([text_feats, img_global, kg_feats], dim=-1)
        logits_base = self.base_expert(base_input)
        
        rel_emb = self.relation_embed(relation_ids)
        kg_rel = torch.cat([kg_feats, rel_emb], dim=-1)
        z_k_tvcs = self.z_k_tvcs_mlp(kg_rel)
        z_k_cls = self.z_k_cls_mlp(kg_feats)
        
        img_patch_proj = self.patch_proj(img_patch)
        
        q = self.Wq(z_k_tvcs)
        k = self.Wk(img_patch_proj)
        v = self.Wv(img_patch_proj)
        
        attn_logits = torch.einsum('bd,bpd->bp', q, k) / (512.0 ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        z_v = torch.einsum('bp,bpd->bd', attn_weights, v)
        
        diff = torch.abs(z_k_tvcs - z_v)
        prod = z_k_tvcs * z_v
        c_input = torch.cat([z_k_tvcs, z_v, diff, prod], dim=-1)
        c_logit = self.c_logit_mlp(c_input).squeeze(-1)
        
        c_emb = self.c_emb_mlp(c_logit.unsqueeze(-1))
        
        tvcs_input = torch.cat([text_feats, img_global, z_k_cls, z_v, c_emb], dim=-1)
        logits_tvcs = self.tvcs_expert(tvcs_input)
        
        gate_input = torch.cat([text_feats, img_global, c_emb], dim=-1)
        g = torch.sigmoid(self.gate_mlp(gate_input))
        g = 0.1 + 0.9 * g
        logits_moe = logits_base + g * (logits_tvcs - logits_base)
        
        ck_boost_input = torch.cat([z_k_tvcs, z_v, c_emb], dim=-1)
        ck_boost = self.ck_boost_mlp(ck_boost_input)
        ck_gate = torch.sigmoid(c_logit).unsqueeze(1)
        beta = 0.5
        logits_final = logits_moe.clone()
        logits_final[:, 2] = logits_final[:, 2] + beta * ck_gate.squeeze(1) * ck_boost.squeeze(1)
        
        return logits_final, logits_base, logits_tvcs, c_logit, g

# --- API Setup ---
app = FastAPI(title="CIKD Model Deployment API")

# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on {device}...")

MODEL_PATH = "checkpoints/cikd/cikd_ckboost_moe_lambda0.7_seed42.pt"
# Checkpoint shapes specify num_relations=1019, kg_dim=100
model = CIKDCKBoostMoE(num_relations=1019, kg_dim=100)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
else:
    print(f"Warning: Model checkpoint not found at {MODEL_PATH}. API will return random predictions unless weights are supplied.")

# --- Schema ---
class FeaturePayload(BaseModel):
    text_feats: List[float]
    img_global: List[float]
    img_patch: List[List[float]]
    kg_feats: List[float]
    relation_id: int

class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: List[float]
    logits: List[float]
    contradiction_prob: float
    gate_value: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "CIKD Model Deployment API is running"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: FeaturePayload):
    try:
        # Convert inputs to tensors and add batch dimension [1, ...]
        t_text = torch.tensor(payload.text_feats, dtype=torch.float32).unsqueeze(0).to(device)
        t_img_g = torch.tensor(payload.img_global, dtype=torch.float32).unsqueeze(0).to(device)
        t_img_p = torch.tensor(payload.img_patch, dtype=torch.float32).unsqueeze(0).to(device)
        t_kg = torch.tensor(payload.kg_feats, dtype=torch.float32).unsqueeze(0).to(device)
        t_rel = torch.tensor([payload.relation_id], dtype=torch.long).to(device)

        # Validate shapes briefly
        if t_text.shape[1] != 768:
            raise HTTPException(status_code=400, detail=f"text_feats must have size 768, got {t_text.shape[1]}")
        if t_img_g.shape[1] != 512:
            raise HTTPException(status_code=400, detail=f"img_global must have size 512, got {t_img_g.shape[1]}")
        if t_img_p.shape[1:] != torch.Size([49, 512]):
            raise HTTPException(status_code=400, detail=f"img_patch must have size [49, 512], got {t_img_p.shape[1:]}")
        if t_kg.shape[1] != 100:
            raise HTTPException(status_code=400, detail=f"kg_feats must have size 100, got {t_kg.shape[1]}")

        with torch.no_grad():
            logits_final, logits_base, logits_tvcs, c_logit, g = model(t_text, t_img_g, t_img_p, t_kg, t_rel)
            
            probs = torch.softmax(logits_final, dim=1).squeeze(0).cpu().numpy().tolist()
            pred_class = int(torch.argmax(logits_final, dim=1).item())
            
            c_prob = torch.sigmoid(c_logit).item()
            gate_val = g.item()

        return PredictResponse(
            predicted_class=pred_class,
            probabilities=probs,
            logits=logits_final.squeeze(0).cpu().numpy().tolist(),
            contradiction_prob=c_prob,
            gate_value=gate_val
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.deploy_api:app", host="0.0.0.0", port=8000, reload=True)
