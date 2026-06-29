"""
CIKD Stage C / D Training Script.
Trains baseline MLP models for Stage C on cached numpy features.
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SimpleMLP(nn.Module):
    """
    Simple MLP Classifier.
    input_dim -> 512 -> dropout 0.2 -> 256 -> dropout 0.2 -> num_classes
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class CIKDResidualMoE(nn.Module):
    """
    CIKD Residual Mixture of Experts (MoE) Model.
    """
    def __init__(self, num_relations, kg_dim=100):
        super().__init__()
        # 1. Base Expert
        # input: concat([text_features, image_features_global, kg_features]) -> dim 1380
        # MLP(1380 -> 512 -> dropout 0.2 -> 256 -> dropout 0.2 -> 6)
        self.base_expert = nn.Sequential(
            nn.Linear(768 + 512 + kg_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        
        # 2. TVCS components same as cikd_light:
        self.relation_embed = nn.Embedding(num_relations, 32)
        
        # z_k_tvcs = MLP([kg_features, relation_embedding]) -> 512
        self.z_k_tvcs_mlp = nn.Sequential(
            nn.Linear(kg_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # z_k_cls = MLP(kg_features) -> 256
        self.z_k_cls_mlp = nn.Sequential(
            nn.Linear(kg_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Project image patch tokens from 512 to 512
        self.patch_proj = nn.Linear(512, 512)
        
        # KG-to-visual attention projections
        self.Wq = nn.Linear(512, 512)
        self.Wk = nn.Linear(512, 512)
        self.Wv = nn.Linear(512, 512)
        
        # Contradiction head MLP
        self.c_logit_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # c_emb MLP
        self.c_emb_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 3. TVCS expert
        # input: concat([text_features, image_features_global, z_k_cls, z_v, c_emb]) -> dim 2112
        # MLP(2112 -> 512 -> dropout 0.2 -> 256 -> dropout 0.2 -> 6)
        self.tvcs_expert = nn.Sequential(
            nn.Linear(768 + 512 + 256 + 512 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        
        # 4. Gate
        # input: concat([text_features, image_features_global, c_emb]) -> dim 1344
        # MLP(gate_input -> 128 -> 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(768 + 512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, text_feats, img_global, img_patch, kg_feats, relation_ids):
        # 1. Base expert logits
        base_input = torch.cat([text_feats, img_global, kg_feats], dim=-1) # [B, 1380]
        logits_base = self.base_expert(base_input) # [B, 6]
        
        # 2. TVCS components
        rel_emb = self.relation_embed(relation_ids) # [B, 32]
        
        # z_k_tvcs
        kg_rel = torch.cat([kg_feats, rel_emb], dim=-1) # [B, kg_dim + 32]
        z_k_tvcs = self.z_k_tvcs_mlp(kg_rel) # [B, 512]
        
        # z_k_cls
        z_k_cls = self.z_k_cls_mlp(kg_feats) # [B, 256]
        
        # project image patch tokens
        img_patch_proj = self.patch_proj(img_patch) # [B, 49, 512]
        
        # KG-to-visual attention
        q = self.Wq(z_k_tvcs) # [B, 512]
        k = self.Wk(img_patch_proj) # [B, 49, 512]
        v = self.Wv(img_patch_proj) # [B, 49, 512]
        
        # attn weights
        attn_logits = torch.einsum('bd,bpd->bp', q, k) / (512.0 ** 0.5) # [B, 49]
        attn_weights = torch.softmax(attn_logits, dim=-1) # [B, 49]
        
        # z_v
        z_v = torch.einsum('bp,bpd->bd', attn_weights, v) # [B, 512]
        
        # contradiction head
        diff = torch.abs(z_k_tvcs - z_v)
        prod = z_k_tvcs * z_v
        c_input = torch.cat([z_k_tvcs, z_v, diff, prod], dim=-1) # [B, 2048]
        c_logit = self.c_logit_mlp(c_input).squeeze(-1) # [B]
        
        # c_emb
        c_emb = self.c_emb_mlp(c_logit.unsqueeze(-1)) # [B, 64]
        
        # 3. TVCS expert logits
        tvcs_input = torch.cat([text_feats, img_global, z_k_cls, z_v, c_emb], dim=-1) # [B, 2112]
        logits_tvcs = self.tvcs_expert(tvcs_input) # [B, 6]
        
        # 4. Gate
        gate_input = torch.cat([text_feats, img_global, c_emb], dim=-1) # [B, 1344]
        g = torch.sigmoid(self.gate_mlp(gate_input)) # [B, 1]
        g = 0.1 + 0.9 * g # [B, 1]
        
        # 5. Residual final logits
        residual = logits_tvcs - logits_base # [B, 6]
        logits_final = logits_base + g * residual # [B, 6]
        
        return logits_final, logits_base, logits_tvcs, c_logit, g


class CIKDCKBoostMoE(nn.Module):
    """
    CIKD CK-Boosted Residual Mixture of Experts (MoE) Model.
    """
    def __init__(self, num_relations, kg_dim=100):
        super().__init__()
        # 1. Base Expert
        # input: concat([text_features, image_features_global, kg_features]) -> dim 1380
        # MLP(1380 -> 512 -> dropout 0.2 -> 256 -> dropout 0.2 -> 6)
        self.base_expert = nn.Sequential(
            nn.Linear(768 + 512 + kg_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        
        # 2. TVCS components same as CIKDResidualMoE:
        self.relation_embed = nn.Embedding(num_relations, 32)
        
        # z_k_tvcs = MLP([kg_features, relation_embedding]) -> 512
        self.z_k_tvcs_mlp = nn.Sequential(
            nn.Linear(kg_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # z_k_cls = MLP(kg_features) -> 256
        self.z_k_cls_mlp = nn.Sequential(
            nn.Linear(kg_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Project image patch tokens from 512 to 512
        self.patch_proj = nn.Linear(512, 512)
        
        # KG-to-visual attention projections
        self.Wq = nn.Linear(512, 512)
        self.Wk = nn.Linear(512, 512)
        self.Wv = nn.Linear(512, 512)
        
        # Contradiction head MLP
        self.c_logit_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # c_emb MLP
        self.c_emb_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 3. TVCS expert
        # input: concat([text_features, image_features_global, z_k_cls, z_v, c_emb]) -> dim 2112
        # MLP(2112 -> 512 -> dropout 0.2 -> 256 -> dropout 0.2 -> 6)
        self.tvcs_expert = nn.Sequential(
            nn.Linear(768 + 512 + 256 + 512 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        
        # 4. Gate
        # input: concat([text_features, image_features_global, c_emb]) -> dim 1344
        # MLP(gate_input -> 128 -> 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(768 + 512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 5. CK-specific boost:
        # ck_boost_input = concat([z_k_tvcs, z_v, c_emb]) -> dim 1088
        # ck_boost = MLP(1088 -> 256 -> dropout 0.2 -> 1)
        self.ck_boost_mlp = nn.Sequential(
            nn.Linear(512 + 512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, text_feats, img_global, img_patch, kg_feats, relation_ids):
        # 1. Base expert logits
        base_input = torch.cat([text_feats, img_global, kg_feats], dim=-1) # [B, 1380]
        logits_base = self.base_expert(base_input) # [B, 6]
        
        # 2. TVCS components
        rel_emb = self.relation_embed(relation_ids) # [B, 32]
        
        # z_k_tvcs
        kg_rel = torch.cat([kg_feats, rel_emb], dim=-1) # [B, kg_dim + 32]
        z_k_tvcs = self.z_k_tvcs_mlp(kg_rel) # [B, 512]
        
        # z_k_cls
        z_k_cls = self.z_k_cls_mlp(kg_feats) # [B, 256]
        
        # project image patch tokens
        img_patch_proj = self.patch_proj(img_patch) # [B, 49, 512]
        
        # KG-to-visual attention
        q = self.Wq(z_k_tvcs) # [B, 512]
        k = self.Wk(img_patch_proj) # [B, 49, 512]
        v = self.Wv(img_patch_proj) # [B, 49, 512]
        
        # attn weights
        attn_logits = torch.einsum('bd,bpd->bp', q, k) / (512.0 ** 0.5) # [B, 49]
        attn_weights = torch.softmax(attn_logits, dim=-1) # [B, 49]
        
        # z_v
        z_v = torch.einsum('bp,bpd->bd', attn_weights, v) # [B, 512]
        
        # contradiction head
        diff = torch.abs(z_k_tvcs - z_v)
        prod = z_k_tvcs * z_v
        c_input = torch.cat([z_k_tvcs, z_v, diff, prod], dim=-1) # [B, 2048]
        c_logit = self.c_logit_mlp(c_input).squeeze(-1) # [B]
        
        # c_emb
        c_emb = self.c_emb_mlp(c_logit.unsqueeze(-1)) # [B, 64]
        
        # 3. TVCS expert logits
        tvcs_input = torch.cat([text_feats, img_global, z_k_cls, z_v, c_emb], dim=-1) # [B, 2112]
        logits_tvcs = self.tvcs_expert(tvcs_input) # [B, 6]
        
        # 4. Gate
        gate_input = torch.cat([text_feats, img_global, c_emb], dim=-1) # [B, 1344]
        g = torch.sigmoid(self.gate_mlp(gate_input)) # [B, 1]
        g = 0.1 + 0.9 * g # [B, 1]
        logits_moe = logits_base + g * (logits_tvcs - logits_base)
        
        # 5. CK-specific boost
        ck_boost_input = torch.cat([z_k_tvcs, z_v, c_emb], dim=-1) # [B, 1088]
        ck_boost = self.ck_boost_mlp(ck_boost_input) # [B, 1]
        ck_gate = torch.sigmoid(c_logit).unsqueeze(1) # [B, 1]
        beta = 0.5
        logits_final = logits_moe.clone()
        logits_final[:, 2] = logits_final[:, 2] + beta * ck_gate.squeeze(1) * ck_boost.squeeze(1)
        
        return logits_final, logits_base, logits_tvcs, c_logit, g

def load_cached_data(cache_dir, subset, stage='C'):
    """Load cached feature arrays for the specified subset."""
    subset_dir = os.path.join(cache_dir, subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"Subset directory does not exist: {subset_dir}")
        
    required_files = [
        'text_features.npy',
        'image_features_global.npy',
        'kg_features.npy',
        'labels_fine.npy',
        'split_ids.npy'
    ]
    if stage.upper() == 'D':
        required_files.extend([
            'image_features_patch.npy',
            'relation_ids.npy',
            'y_ck.npy',
            'sample_ids.npy'
        ])
    
    for f in required_files:
        path = os.path.join(subset_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required cached file not found: {path}")
            
    print(f"Loading cached arrays from: {subset_dir}")
    data = {
        'text_features': np.load(os.path.join(subset_dir, 'text_features.npy'), mmap_mode='r'),
        'image_features_global': np.load(os.path.join(subset_dir, 'image_features_global.npy'), mmap_mode='r'),
        'kg_features': np.load(os.path.join(subset_dir, 'kg_features.npy'), mmap_mode='r'),
        'labels_fine': np.load(os.path.join(subset_dir, 'labels_fine.npy'), mmap_mode='r'),
        'split_ids': np.load(os.path.join(subset_dir, 'split_ids.npy'), mmap_mode='r')
    }
    if stage.upper() == 'D':
        data['image_features_patch'] = np.load(os.path.join(subset_dir, 'image_features_patch.npy'), mmap_mode='r')
        data['relation_ids'] = np.load(os.path.join(subset_dir, 'relation_ids.npy'), mmap_mode='r')
        data['y_ck'] = np.load(os.path.join(subset_dir, 'y_ck.npy'), mmap_mode='r')
        data['sample_ids'] = np.load(os.path.join(subset_dir, 'sample_ids.npy'), mmap_mode='r')
    return data

def train_baseline(model_name, features, labels, split_ids, num_classes, args, device):
    """Train a single baseline model."""
    print("\n" + "=" * 80)
    print(f"Training Model: {model_name} (Seed: {args.seed})")
    print("=" * 80)
    
    # Extract train, validation, and test splits
    train_mask = (split_ids == 0)
    val_mask = (split_ids == 1)
    
    X_train = np.array(features[train_mask])
    y_train = np.array(labels[train_mask])
    X_val = np.array(features[val_mask])
    y_val = np.array(labels[val_mask])
    
    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")
    
    # Compute class-weighted loss weights from train labels
    counts = np.bincount(y_train, minlength=num_classes)
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = len(y_train) / (num_classes * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class counts in train: {counts.tolist()}")
    print(f"Calculated class weights: {weights.tolist()}")
    
    # Construct PyTorch DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Model, Optimizer, and Loss
    model = SimpleMLP(input_dim=features.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    checkpoint_dir = "checkpoints/baselines"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_seed{args.seed}.pt")
    
    best_score = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    patience = 5
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(bx)
            
        train_loss = train_loss_sum / len(train_dataset)
        
        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(by.numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Compute metrics
        acc = accuracy_score(val_targets, val_preds)
        macro_f1 = f1_score(val_targets, val_preds, average='macro')
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
        ck_f1 = per_class_f1[2]
        
        sel_score = 0.5 * macro_f1 + 0.5 * ck_f1
        
        # Format printing: model name, epoch, train loss, val macro_f1, val ck_f1, selection score
        print(f"{model_name}, epoch {epoch+1}, loss {train_loss:.4f}, val macro_f1 {macro_f1:.4f}, val ck_f1 {ck_f1:.4f}, selection score {sel_score:.4f}")
        
        if sel_score > best_score:
            best_score = sel_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            # Save best checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'selection_score': best_score,
                'val_metrics': {
                    'accuracy': acc,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'ck_f1': ck_f1,
                    'per_class_f1': per_class_f1.tolist()
                }
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with selection score {best_score:.4f}.")
                break
                
    # Load best checkpoint for final validation evaluation
    print(f"Loading best checkpoint from {checkpoint_path} for final validation evaluation...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(by.numpy())
            
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    acc = accuracy_score(val_targets, val_preds)
    macro_f1 = f1_score(val_targets, val_preds, average='macro')
    weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
    per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
    ck_f1 = per_class_f1[2]
    
    print(f"Finished Training. Final Val Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | CK-F1: {ck_f1:.4f}")
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1,
        'per_class_f1': per_class_f1
    }

class CIKDLight(nn.Module):
    """
    CIKD-Light Model Architecture.
    """
    def __init__(self, num_relations, kg_dim=100):
        super().__init__()
        # Relation embedding
        self.relation_embed = nn.Embedding(num_relations, 32)
        
        # z_k_tvcs = MLP([kg_features, relation_embedding]) -> 512
        # Input size: kg_dim + 32
        self.z_k_tvcs_mlp = nn.Sequential(
            nn.Linear(kg_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # z_k_cls = MLP(kg_features) -> 256
        # Input size: kg_dim
        self.z_k_cls_mlp = nn.Sequential(
            nn.Linear(kg_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Project image patch tokens from 512 to 512
        self.patch_proj = nn.Linear(512, 512)
        
        # KG-to-visual attention projections
        self.Wq = nn.Linear(512, 512)
        self.Wk = nn.Linear(512, 512)
        self.Wv = nn.Linear(512, 512)
        
        # Contradiction head MLP
        # Input size: 512 * 4 = 2048
        self.c_logit_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # c_emb MLP
        # Input size: 1
        self.c_emb_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Classifier
        # Input size: 768 (text) + 512 (global img) + 256 (z_k_cls) + 512 (z_v) + 64 (c_emb) = 2112
        self.classifier = nn.Sequential(
            nn.Linear(2112, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        
    def forward(self, text_feats, img_global, img_patch, kg_feats, relation_ids):
        # relation embedding
        rel_emb = self.relation_embed(relation_ids) # [B, 32]
        
        # z_k_tvcs
        kg_rel = torch.cat([kg_feats, rel_emb], dim=-1) # [B, kg_dim + 32]
        z_k_tvcs = self.z_k_tvcs_mlp(kg_rel) # [B, 512]
        
        # z_k_cls
        z_k_cls = self.z_k_cls_mlp(kg_feats) # [B, 256]
        
        # project image patch tokens
        img_patch_proj = self.patch_proj(img_patch) # [B, 49, 512]
        
        # KG-to-visual attention
        q = self.Wq(z_k_tvcs) # [B, 512]
        k = self.Wk(img_patch_proj) # [B, 49, 512]
        v = self.Wv(img_patch_proj) # [B, 49, 512]
        
        # attn weights
        # query dot keys / sqrt(512)
        attn_logits = torch.einsum('bd,bpd->bp', q, k) / (512.0 ** 0.5) # [B, 49]
        attn_weights = torch.softmax(attn_logits, dim=-1) # [B, 49]
        
        # z_v: weighted sum of values
        z_v = torch.einsum('bp,bpd->bd', attn_weights, v) # [B, 512]
        
        # contradiction head
        diff = torch.abs(z_k_tvcs - z_v)
        prod = z_k_tvcs * z_v
        c_input = torch.cat([z_k_tvcs, z_v, diff, prod], dim=-1) # [B, 2048]
        c_logit = self.c_logit_mlp(c_input).squeeze(-1) # [B]
        
        # c_emb
        c_emb = self.c_emb_mlp(c_logit.unsqueeze(-1)) # [B, 64]
        
        # classifier
        cls_input = torch.cat([text_feats, img_global, z_k_cls, z_v, c_emb], dim=-1) # [B, 2112]
        logits = self.classifier(cls_input) # [B, 6]
        
        return logits, c_logit

def train_cikd_light(data, num_classes, args, device):
    """Train CIKD-Light model on cached features."""
    print("\n" + "=" * 80)
    print(f"Training CIKD-Light (Seed: {args.seed}, Lambda TVCS: {args.lambda_tvcs})")
    print("=" * 80)
    
    # Extract train and validation splits
    train_mask = (data['split_ids'] == 0)
    val_mask = (data['split_ids'] == 1)
    
    # Train tensors
    train_text = torch.tensor(data['text_features'][train_mask], dtype=torch.float32)
    train_img_global = torch.tensor(data['image_features_global'][train_mask], dtype=torch.float32)
    train_img_patch = torch.tensor(data['image_features_patch'][train_mask], dtype=torch.float32)
    train_kg = torch.tensor(data['kg_features'][train_mask], dtype=torch.float32)
    train_rel_ids = torch.tensor(data['relation_ids'][train_mask], dtype=torch.long)
    train_labels = torch.tensor(data['labels_fine'][train_mask], dtype=torch.long)
    train_y_ck = torch.tensor(data['y_ck'][train_mask], dtype=torch.float32)
    train_sample_ids = torch.tensor(data['sample_ids'][train_mask], dtype=torch.long)
    
    # Val tensors
    val_text = torch.tensor(data['text_features'][val_mask], dtype=torch.float32)
    val_img_global = torch.tensor(data['image_features_global'][val_mask], dtype=torch.float32)
    val_img_patch = torch.tensor(data['image_features_patch'][val_mask], dtype=torch.float32)
    val_kg = torch.tensor(data['kg_features'][val_mask], dtype=torch.float32)
    val_rel_ids = torch.tensor(data['relation_ids'][val_mask], dtype=torch.long)
    val_labels = torch.tensor(data['labels_fine'][val_mask], dtype=torch.long)
    val_y_ck = torch.tensor(data['y_ck'][val_mask], dtype=torch.float32)
    val_sample_ids = torch.tensor(data['sample_ids'][val_mask], dtype=torch.long)
    
    print(f"Train set shape: {train_text.shape}, Val set shape: {val_text.shape}")
    
    # Compute class-weighted loss weights from train labels
    counts = np.bincount(data['labels_fine'][train_mask], minlength=num_classes)
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = len(train_labels) / (num_classes * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class counts in train: {counts.tolist()}")
    print(f"Calculated class weights: {weights.tolist()}")
    
    # Construct PyTorch DataLoaders
    train_dataset = TensorDataset(
        train_text, train_img_global, train_img_patch, train_kg,
        train_rel_ids, train_labels, train_y_ck, train_sample_ids
    )
    val_dataset = TensorDataset(
        val_text, val_img_global, val_img_patch, val_kg,
        val_rel_ids, val_labels, val_y_ck, val_sample_ids
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    num_relations = int(data['relation_ids'].max()) + 1
    model = CIKDLight(num_relations=num_relations, kg_dim=data['kg_features'].shape[1]).to(device)
    
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_tvcs = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    checkpoint_dir = "checkpoints/cikd"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"cikd_light_lambda{args.lambda_tvcs}_seed{args.seed}.pt")
    
    best_score = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    patience = 5
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        train_cls_loss_sum = 0.0
        train_tvcs_loss_sum = 0.0
        train_total_loss_sum = 0.0
        train_tvcs_count = 0
        train_count = 0
        
        for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, _ in train_loader:
            bx_text = bx_text.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_img_patch = bx_img_patch.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_label = bx_label.to(device)
            bx_y_ck = bx_y_ck.to(device)
            
            optimizer.zero_grad()
            logits, c_logits = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
            
            # L_cls
            loss_cls = criterion_cls(logits, bx_label)
            
            # L_tvcs
            mask = (bx_y_ck != -1)
            if mask.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
            else:
                loss_tvcs = torch.tensor(0.0, device=device)
                
            # total loss
            loss_total = loss_cls + args.lambda_tvcs * loss_tvcs
            
            loss_total.backward()
            
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # accumulate losses
            train_cls_loss_sum += loss_cls.item() * len(bx_label)
            train_count += len(bx_label)
            if mask.sum() > 0:
                train_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                train_tvcs_count += mask.sum().item()
            train_total_loss_sum += loss_total.item() * len(bx_label)
            
        train_cls_loss = train_cls_loss_sum / train_count
        train_tvcs_loss = train_tvcs_loss_sum / train_tvcs_count if train_tvcs_count > 0 else 0.0
        train_total_loss = train_total_loss_sum / train_count
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_c_logits = []
        val_y_ck_list = []
        
        val_cls_loss_sum = 0.0
        val_tvcs_loss_sum = 0.0
        val_tvcs_count = 0
        val_count = 0
        
        with torch.no_grad():
            for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, _ in val_loader:
                bx_text = bx_text.to(device)
                bx_img_global = bx_img_global.to(device)
                bx_img_patch = bx_img_patch.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)
                bx_label = bx_label.to(device)
                bx_y_ck = bx_y_ck.to(device)
                
                logits, c_logits = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
                
                loss_cls = criterion_cls(logits, bx_label)
                val_cls_loss_sum += loss_cls.item() * len(bx_label)
                val_count += len(bx_label)
                
                mask = (bx_y_ck != -1)
                if mask.sum() > 0:
                    loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
                    val_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                    val_tvcs_count += mask.sum().item()
                    
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(bx_label.cpu().numpy())
                val_c_logits.extend(c_logits.cpu().numpy())
                val_y_ck_list.extend(bx_y_ck.cpu().numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_c_logits = np.array(val_c_logits)
        val_c_probs = 1.0 / (1.0 + np.exp(-val_c_logits))
        val_y_ck_arr = np.array(val_y_ck_list)
        
        val_cls_loss = val_cls_loss_sum / val_count
        val_tvcs_loss = val_tvcs_loss_sum / val_tvcs_count if val_tvcs_count > 0 else 0.0
        val_total_loss = val_cls_loss + args.lambda_tvcs * val_tvcs_loss
        
        acc = accuracy_score(val_targets, val_preds)
        macro_f1 = f1_score(val_targets, val_preds, average='macro')
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
        ck_f1 = per_class_f1[2]
        sel_score = 0.5 * macro_f1 + 0.5 * ck_f1
        
        # TVCS metrics
        tvcs_mask = (val_y_ck_arr != -1)
        val_y_ck_tvcs = val_y_ck_arr[tvcs_mask]
        val_c_probs_tvcs = val_c_probs[tvcs_mask]
        
        if len(np.unique(val_y_ck_tvcs)) > 1:
            from sklearn.metrics import roc_auc_score
            tvcs_auc_ck_vs_real = roc_auc_score(val_y_ck_tvcs, val_c_probs_tvcs)
        else:
            tvcs_auc_ck_vs_real = 0.5
            
        real_mask = (val_y_ck_arr == 0)
        mean_c_real = float(np.mean(val_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
        
        ck_mask_y = (val_y_ck_arr == 1)
        mean_c_ck = float(np.mean(val_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
        
        tvcs_delta = mean_c_ck - mean_c_real
        
        # Print metrics
        print(f"cikd_light, epoch {epoch+1}:")
        print(f"  accuracy: {acc:.4f}")
        print(f"  macro_f1: {macro_f1:.4f}")
        print(f"  weighted_f1: {weighted_f1:.4f}")
        print(f"  ck_f1: {ck_f1:.4f}")
        print(f"  selection_score: {sel_score:.4f}")
        print(f"  tvcs_auc_ck_vs_real: {tvcs_auc_ck_vs_real:.4f}")
        print(f"  mean_c_real: {mean_c_real:.4f}")
        print(f"  mean_c_ck: {mean_c_ck:.4f}")
        print(f"  tvcs_delta: {tvcs_delta:.4f}")
        print(f"  cls_loss: {val_cls_loss:.4f}")
        print(f"  tvcs_loss: {val_tvcs_loss:.4f}")
        print(f"  total_loss: {val_total_loss:.4f}")
        
        if sel_score > best_score:
            best_score = sel_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'selection_score': best_score,
                'val_metrics': {
                    'accuracy': acc,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'ck_f1': ck_f1,
                    'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
                    'mean_c_real': mean_c_real,
                    'mean_c_ck': mean_c_ck,
                    'tvcs_delta': tvcs_delta,
                    'cls_loss': val_cls_loss,
                    'tvcs_loss': val_tvcs_loss,
                    'total_loss': val_total_loss,
                    'per_class_f1': per_class_f1.tolist()
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with selection score {best_score:.4f}.")
                break
                
    # Load best checkpoint for final evaluation
    print(f"Loading best checkpoint from {checkpoint_path} for final validation evaluation...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    val_preds = []
    val_targets = []
    val_c_logits = []
    val_y_ck_list = []
    val_sample_ids_list = []
    
    val_cls_loss_sum = 0.0
    val_tvcs_loss_sum = 0.0
    val_tvcs_count = 0
    val_count = 0
    
    with torch.no_grad():
        for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, bx_sample_id in val_loader:
            bx_text = bx_text.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_img_patch = bx_img_patch.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_label = bx_label.to(device)
            bx_y_ck = bx_y_ck.to(device)
            
            logits, c_logits = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
            
            loss_cls = criterion_cls(logits, bx_label)
            val_cls_loss_sum += loss_cls.item() * len(bx_label)
            val_count += len(bx_label)
            
            mask = (bx_y_ck != -1)
            if mask.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
                val_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                val_tvcs_count += mask.sum().item()
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(bx_label.cpu().numpy())
            val_c_logits.extend(c_logits.cpu().numpy())
            val_y_ck_list.extend(bx_y_ck.cpu().numpy())
            val_sample_ids_list.extend(bx_sample_id.cpu().numpy())
            
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_c_logits = np.array(val_c_logits)
    val_c_probs = 1.0 / (1.0 + np.exp(-val_c_logits))
    val_y_ck_arr = np.array(val_y_ck_list)
    val_sample_ids_arr = np.array(val_sample_ids_list)
    
    val_cls_loss = val_cls_loss_sum / val_count
    val_tvcs_loss = val_tvcs_loss_sum / val_tvcs_count if val_tvcs_count > 0 else 0.0
    val_total_loss = val_cls_loss + args.lambda_tvcs * val_tvcs_loss
    
    acc = accuracy_score(val_targets, val_preds)
    macro_f1 = f1_score(val_targets, val_preds, average='macro')
    weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
    per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
    ck_f1 = per_class_f1[2]
    sel_score = 0.5 * macro_f1 + 0.5 * ck_f1
    
    # TVCS metrics
    tvcs_mask = (val_y_ck_arr != -1)
    val_y_ck_tvcs = val_y_ck_arr[tvcs_mask]
    val_c_probs_tvcs = val_c_probs[tvcs_mask]
    
    if len(np.unique(val_y_ck_tvcs)) > 1:
        from sklearn.metrics import roc_auc_score
        tvcs_auc_ck_vs_real = roc_auc_score(val_y_ck_tvcs, val_c_probs_tvcs)
    else:
        tvcs_auc_ck_vs_real = 0.5
        
    real_mask = (val_y_ck_arr == 0)
    mean_c_real = float(np.mean(val_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
    
    ck_mask_y = (val_y_ck_arr == 1)
    mean_c_ck = float(np.mean(val_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
    
    tvcs_delta = mean_c_ck - mean_c_real
    
    print(f"\nFinished Training. Final Best Val Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | CK-F1: {ck_f1:.4f}")
    
    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 03_cikd_metrics_val.csv
    cikd_metrics = [{
        'model': 'cikd_light',
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1,
        'selection_score': sel_score,
        'cls_loss': val_cls_loss,
        'total_loss': val_total_loss
    }]
    df_cikd_metrics = pd.DataFrame(cikd_metrics)
    cikd_metrics_path = os.path.join(args.out_dir, '03_cikd_metrics_val.csv')
    df_cikd_metrics.to_csv(cikd_metrics_path, index=False)
    print(f"Saved CIKD metrics to: {cikd_metrics_path}")
    
    # 2. 03_tvcs_metrics_val.csv
    tvcs_metrics = [{
        'model': 'cikd_light',
        'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
        'mean_c_real': mean_c_real,
        'mean_c_ck': mean_c_ck,
        'tvcs_delta': tvcs_delta,
        'tvcs_loss': val_tvcs_loss
    }]
    df_tvcs_metrics = pd.DataFrame(tvcs_metrics)
    tvcs_metrics_path = os.path.join(args.out_dir, '03_tvcs_metrics_val.csv')
    df_tvcs_metrics.to_csv(tvcs_metrics_path, index=False)
    print(f"Saved TVCS metrics to: {tvcs_metrics_path}")
    
    # 3. 03_per_class_f1_val.csv
    per_class_row = {'model': 'cikd_light'}
    for c in range(num_classes):
        per_class_row[f'f1_class_{c}'] = per_class_f1[c]
    df_per_class = pd.DataFrame([per_class_row])
    per_class_path = os.path.join(args.out_dir, '03_per_class_f1_val.csv')
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved per-class F1 to: {per_class_path}")
    
    # 4. 03_tvcs_scores_best_val.csv
    df_tvcs_scores = pd.DataFrame({
        'sample_id': val_sample_ids_arr,
        'y_ck': val_y_ck_arr,
        'c_logit': val_c_logits,
        'c_score': val_c_probs
    })
    tvcs_scores_path = os.path.join(args.out_dir, '03_tvcs_scores_best_val.csv')
    df_tvcs_scores.to_csv(tvcs_scores_path, index=False)
    print(f"Saved best TVCS scores to: {tvcs_scores_path}")
    
    # 5. 03_tvcs_hist_best.png
    plt.figure(figsize=(8, 6))
    real_scores = val_c_probs[val_y_ck_arr == 0]
    ck_scores = val_c_probs[val_y_ck_arr == 1]
    
    plt.hist(real_scores, bins=30, alpha=0.5, label=f'Real (y_ck=0, mean={mean_c_real:.4f})', color='green', edgecolor='k')
    plt.hist(ck_scores, bins=30, alpha=0.5, label=f'Contradictory (y_ck=1, mean={mean_c_ck:.4f})', color='red', edgecolor='k')
    
    plt.xlabel('Contradiction Score (c_score)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Contradiction Scores (Val Set, tvcs_delta={tvcs_delta:.4f})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    hist_path = os.path.join(args.out_dir, '03_tvcs_hist_best.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved TVCS scores histogram to: {hist_path}")

def train_cikd_residual_moe(data, num_classes, args, device):
    """Train CIKD-Residual MoE model on cached features."""
    print("\n" + "=" * 80)
    print(f"Training CIKD-Residual MoE (Seed: {args.seed}, Lambda TVCS: {args.lambda_tvcs})")
    print("=" * 80)
    
    # Set seed before training to ensure reproducibility
    set_seed(args.seed)
    
    # Extract train and validation splits
    train_mask = (data['split_ids'] == 0)
    val_mask = (data['split_ids'] == 1)
    
    # Train tensors
    train_text = torch.tensor(data['text_features'][train_mask], dtype=torch.float32)
    train_img_global = torch.tensor(data['image_features_global'][train_mask], dtype=torch.float32)
    train_img_patch = torch.tensor(data['image_features_patch'][train_mask], dtype=torch.float32)
    train_kg = torch.tensor(data['kg_features'][train_mask], dtype=torch.float32)
    train_rel_ids = torch.tensor(data['relation_ids'][train_mask], dtype=torch.long)
    train_labels = torch.tensor(data['labels_fine'][train_mask], dtype=torch.long)
    train_y_ck = torch.tensor(data['y_ck'][train_mask], dtype=torch.float32)
    train_sample_ids = torch.tensor(data['sample_ids'][train_mask], dtype=torch.long)
    
    # Val tensors
    val_text = torch.tensor(data['text_features'][val_mask], dtype=torch.float32)
    val_img_global = torch.tensor(data['image_features_global'][val_mask], dtype=torch.float32)
    val_img_patch = torch.tensor(data['image_features_patch'][val_mask], dtype=torch.float32)
    val_kg = torch.tensor(data['kg_features'][val_mask], dtype=torch.float32)
    val_rel_ids = torch.tensor(data['relation_ids'][val_mask], dtype=torch.long)
    val_labels = torch.tensor(data['labels_fine'][val_mask], dtype=torch.long)
    val_y_ck = torch.tensor(data['y_ck'][val_mask], dtype=torch.float32)
    val_sample_ids = torch.tensor(data['sample_ids'][val_mask], dtype=torch.long)
    
    print(f"Train set shape: {train_text.shape}, Val set shape: {val_text.shape}")
    
    # Compute class-weighted loss weights from train labels
    counts = np.bincount(data['labels_fine'][train_mask], minlength=num_classes)
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = len(train_labels) / (num_classes * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class counts in train: {counts.tolist()}")
    print(f"Calculated class weights: {weights.tolist()}")
    
    # Construct PyTorch DataLoaders
    train_dataset = TensorDataset(
        train_text, train_img_global, train_img_patch, train_kg,
        train_rel_ids, train_labels, train_y_ck, train_sample_ids
    )
    val_dataset = TensorDataset(
        val_text, val_img_global, val_img_patch, val_kg,
        val_rel_ids, val_labels, val_y_ck, val_sample_ids
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    num_relations = int(data['relation_ids'].max()) + 1
    model = CIKDResidualMoE(num_relations=num_relations, kg_dim=data['kg_features'].shape[1]).to(device)
    
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_tvcs = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    checkpoint_dir = "checkpoints/cikd"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"cikd_residual_moe_lambda{args.lambda_tvcs}_seed{args.seed}.pt")
    
    best_score = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    patience = 5
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        train_cls_loss_sum = 0.0
        train_base_loss_sum = 0.0
        train_tvcs_loss_sum = 0.0
        train_total_loss_sum = 0.0
        train_tvcs_count = 0
        train_count = 0
        
        for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, _ in train_loader:
            bx_text = bx_text.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_img_patch = bx_img_patch.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_label = bx_label.to(device)
            bx_y_ck = bx_y_ck.to(device)
            
            optimizer.zero_grad()
            logits_final, logits_base, logits_tvcs, c_logits, g = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
            
            # L_cls
            loss_cls = criterion_cls(logits_final, bx_label)
            
            # L_base
            loss_base = criterion_cls(logits_base, bx_label)
            
            # L_tvcs
            mask = (bx_y_ck != -1)
            if mask.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
            else:
                loss_tvcs = torch.tensor(0.0, device=device)
                
            # total loss
            loss_total = loss_cls + 0.3 * loss_base + args.lambda_tvcs * loss_tvcs
            
            loss_total.backward()
            
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # accumulate losses
            train_cls_loss_sum += loss_cls.item() * len(bx_label)
            train_base_loss_sum += loss_base.item() * len(bx_label)
            train_count += len(bx_label)
            if mask.sum() > 0:
                train_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                train_tvcs_count += mask.sum().item()
            train_total_loss_sum += loss_total.item() * len(bx_label)
            
        train_cls_loss = train_cls_loss_sum / train_count
        train_base_loss = train_base_loss_sum / train_count
        train_tvcs_loss = train_tvcs_loss_sum / train_tvcs_count if train_tvcs_count > 0 else 0.0
        train_total_loss = train_total_loss_sum / train_count
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_c_logits = []
        val_y_ck_list = []
        
        val_cls_loss_sum = 0.0
        val_base_loss_sum = 0.0
        val_tvcs_loss_sum = 0.0
        val_tvcs_count = 0
        val_count = 0
        
        with torch.no_grad():
            for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, _ in val_loader:
                bx_text = bx_text.to(device)
                bx_img_global = bx_img_global.to(device)
                bx_img_patch = bx_img_patch.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)
                bx_label = bx_label.to(device)
                bx_y_ck = bx_y_ck.to(device)
                
                logits_final, logits_base, logits_tvcs, c_logits, g = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
                
                loss_cls = criterion_cls(logits_final, bx_label)
                val_cls_loss_sum += loss_cls.item() * len(bx_label)
                
                loss_base = criterion_cls(logits_base, bx_label)
                val_base_loss_sum += loss_base.item() * len(bx_label)
                
                val_count += len(bx_label)
                
                mask = (bx_y_ck != -1)
                if mask.sum() > 0:
                    loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
                    val_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                    val_tvcs_count += mask.sum().item()
                    
                preds = torch.argmax(logits_final, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(bx_label.cpu().numpy())
                val_c_logits.extend(c_logits.cpu().numpy())
                val_y_ck_list.extend(bx_y_ck.cpu().numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_c_logits = np.array(val_c_logits)
        val_c_probs = 1.0 / (1.0 + np.exp(-val_c_logits))
        val_y_ck_arr = np.array(val_y_ck_list)
        
        val_cls_loss = val_cls_loss_sum / val_count
        val_base_loss = val_base_loss_sum / val_count
        val_tvcs_loss = val_tvcs_loss_sum / val_tvcs_count if val_tvcs_count > 0 else 0.0
        val_total_loss = val_cls_loss + 0.3 * val_base_loss + args.lambda_tvcs * val_tvcs_loss
        
        acc = accuracy_score(val_targets, val_preds)
        macro_f1 = f1_score(val_targets, val_preds, average='macro')
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
        ck_f1 = per_class_f1[2]
        sel_score = 0.5 * macro_f1 + 0.5 * ck_f1
        
        # TVCS metrics
        tvcs_mask = (val_y_ck_arr != -1)
        val_y_ck_tvcs = val_y_ck_arr[tvcs_mask]
        val_c_probs_tvcs = val_c_probs[tvcs_mask]
        
        if len(np.unique(val_y_ck_tvcs)) > 1:
            from sklearn.metrics import roc_auc_score
            tvcs_auc_ck_vs_real = roc_auc_score(val_y_ck_tvcs, val_c_probs_tvcs)
        else:
            tvcs_auc_ck_vs_real = 0.5
            
        real_mask = (val_y_ck_arr == 0)
        mean_c_real = float(np.mean(val_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
        
        ck_mask_y = (val_y_ck_arr == 1)
        mean_c_ck = float(np.mean(val_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
        
        tvcs_delta = mean_c_ck - mean_c_real
        
        # Print metrics
        print(f"cikd_residual_moe, epoch {epoch+1}:")
        print(f"  accuracy: {acc:.4f}")
        print(f"  macro_f1: {macro_f1:.4f}")
        print(f"  weighted_f1: {weighted_f1:.4f}")
        print(f"  ck_f1: {ck_f1:.4f}")
        print(f"  selection_score: {sel_score:.4f}")
        print(f"  tvcs_auc_ck_vs_real: {tvcs_auc_ck_vs_real:.4f}")
        print(f"  mean_c_real: {mean_c_real:.4f}")
        print(f"  mean_c_ck: {mean_c_ck:.4f}")
        print(f"  tvcs_delta: {tvcs_delta:.4f}")
        print(f"  cls_loss: {val_cls_loss:.4f}")
        print(f"  tvcs_loss: {val_tvcs_loss:.4f}")
        print(f"  total_loss: {val_total_loss:.4f}")
        
        if sel_score > best_score:
            best_score = sel_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'selection_score': best_score,
                'val_metrics': {
                    'accuracy': acc,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'ck_f1': ck_f1,
                    'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
                    'mean_c_real': mean_c_real,
                    'mean_c_ck': mean_c_ck,
                    'tvcs_delta': tvcs_delta,
                    'cls_loss': val_cls_loss,
                    'tvcs_loss': val_tvcs_loss,
                    'total_loss': val_total_loss,
                    'per_class_f1': per_class_f1.tolist()
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with selection score {best_score:.4f}.")
                break
                
    # Load best checkpoint for final evaluation
    print(f"Loading best checkpoint from {checkpoint_path} for final validation evaluation...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    val_preds = []
    val_targets = []
    val_c_logits = []
    val_y_ck_list = []
    val_sample_ids_list = []
    
    val_cls_loss_sum = 0.0
    val_base_loss_sum = 0.0
    val_tvcs_loss_sum = 0.0
    val_tvcs_count = 0
    val_count = 0
    
    with torch.no_grad():
        for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, bx_sample_id in val_loader:
            bx_text = bx_text.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_img_patch = bx_img_patch.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_label = bx_label.to(device)
            bx_y_ck = bx_y_ck.to(device)
            
            logits_final, logits_base, logits_tvcs, c_logits, g = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
            
            loss_cls = criterion_cls(logits_final, bx_label)
            val_cls_loss_sum += loss_cls.item() * len(bx_label)
            
            loss_base = criterion_cls(logits_base, bx_label)
            val_base_loss_sum += loss_base.item() * len(bx_label)
            
            val_count += len(bx_label)
            
            mask = (bx_y_ck != -1)
            if mask.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
                val_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                val_tvcs_count += mask.sum().item()
                
            preds = torch.argmax(logits_final, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(bx_label.cpu().numpy())
            val_c_logits.extend(c_logits.cpu().numpy())
            val_y_ck_list.extend(bx_y_ck.cpu().numpy())
            val_sample_ids_list.extend(bx_sample_id.cpu().numpy())
            
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_c_logits = np.array(val_c_logits)
    val_c_probs = 1.0 / (1.0 + np.exp(-val_c_logits))
    val_y_ck_arr = np.array(val_y_ck_list)
    val_sample_ids_arr = np.array(val_sample_ids_list)
    
    val_cls_loss = val_cls_loss_sum / val_count
    val_base_loss = val_base_loss_sum / val_count
    val_tvcs_loss = val_tvcs_loss_sum / val_tvcs_count if val_tvcs_count > 0 else 0.0
    val_total_loss = val_cls_loss + 0.3 * val_base_loss + args.lambda_tvcs * val_tvcs_loss
    
    acc = accuracy_score(val_targets, val_preds)
    macro_f1 = f1_score(val_targets, val_preds, average='macro')
    weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
    per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
    ck_f1 = per_class_f1[2]
    sel_score = 0.5 * macro_f1 + 0.5 * ck_f1
    
    # TVCS metrics
    tvcs_mask = (val_y_ck_arr != -1)
    val_y_ck_tvcs = val_y_ck_arr[tvcs_mask]
    val_c_probs_tvcs = val_c_probs[tvcs_mask]
    
    if len(np.unique(val_y_ck_tvcs)) > 1:
        from sklearn.metrics import roc_auc_score
        tvcs_auc_ck_vs_real = roc_auc_score(val_y_ck_tvcs, val_c_probs_tvcs)
    else:
        tvcs_auc_ck_vs_real = 0.5
        
    real_mask = (val_y_ck_arr == 0)
    mean_c_real = float(np.mean(val_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
    
    ck_mask_y = (val_y_ck_arr == 1)
    mean_c_ck = float(np.mean(val_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
    
    tvcs_delta = mean_c_ck - mean_c_real
    
    print(f"\nFinished Training. Final Best Val Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | CK-F1: {ck_f1:.4f}")
    
    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 03_cikd_metrics_val.csv
    cikd_metrics = [{
        'model': 'cikd_residual_moe',
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1,
        'selection_score': sel_score,
        'cls_loss': val_cls_loss,
        'total_loss': val_total_loss
    }]
    df_cikd_metrics = pd.DataFrame(cikd_metrics)
    cikd_metrics_path = os.path.join(args.out_dir, '03_cikd_metrics_val.csv')
    df_cikd_metrics.to_csv(cikd_metrics_path, index=False)
    print(f"Saved CIKD metrics to: {cikd_metrics_path}")
    
    # 2. 03_tvcs_metrics_val.csv
    tvcs_metrics = [{
        'model': 'cikd_residual_moe',
        'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
        'mean_c_real': mean_c_real,
        'mean_c_ck': mean_c_ck,
        'tvcs_delta': tvcs_delta,
        'tvcs_loss': val_tvcs_loss
    }]
    df_tvcs_metrics = pd.DataFrame(tvcs_metrics)
    tvcs_metrics_path = os.path.join(args.out_dir, '03_tvcs_metrics_val.csv')
    df_tvcs_metrics.to_csv(tvcs_metrics_path, index=False)
    print(f"Saved TVCS metrics to: {tvcs_metrics_path}")
    
    # 3. 03_per_class_f1_val.csv
    per_class_row = {'model': 'cikd_residual_moe'}
    for c in range(num_classes):
        per_class_row[f'f1_class_{c}'] = per_class_f1[c]
    df_per_class = pd.DataFrame([per_class_row])
    per_class_path = os.path.join(args.out_dir, '03_per_class_f1_val.csv')
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved per-class F1 to: {per_class_path}")
    
    # 4. 03_tvcs_scores_best_val.csv
    df_tvcs_scores = pd.DataFrame({
        'sample_id': val_sample_ids_arr,
        'y_ck': val_y_ck_arr,
        'c_logit': val_c_logits,
        'c_score': val_c_probs
    })
    tvcs_scores_path = os.path.join(args.out_dir, '03_tvcs_scores_best_val.csv')
    df_tvcs_scores.to_csv(tvcs_scores_path, index=False)
    print(f"Saved best TVCS scores to: {tvcs_scores_path}")
    
    # 5. 03_tvcs_hist_best.png
    plt.figure(figsize=(8, 6))
    real_scores = val_c_probs[val_y_ck_arr == 0]
    ck_scores = val_c_probs[val_y_ck_arr == 1]
    
    plt.hist(real_scores, bins=30, alpha=0.5, label=f'Real (y_ck=0, mean={mean_c_real:.4f})', color='green', edgecolor='k')
    plt.hist(ck_scores, bins=30, alpha=0.5, label=f'Contradictory (y_ck=1, mean={mean_c_ck:.4f})', color='red', edgecolor='k')
    
    plt.xlabel('Contradiction Score (c_score)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Contradiction Scores (Val Set, tvcs_delta={tvcs_delta:.4f})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    hist_path = os.path.join(args.out_dir, '03_tvcs_hist_best.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved TVCS scores histogram to: {hist_path}")


def train_cikd_ckboost_moe(data, num_classes, args, device):
    """Train CIKD-CKBoost MoE model on cached features."""
    print("\n" + "=" * 80)
    print(f"Training CIKD-CKBoost MoE (Seed: {args.seed}, Lambda TVCS: {args.lambda_tvcs})")
    print("=" * 80)
    
    # Set seed before training to ensure reproducibility
    set_seed(args.seed)
    
    # Extract train and validation splits
    train_mask = (data['split_ids'] == 0)
    val_mask = (data['split_ids'] == 1)
    
    # Train tensors
    train_text = torch.tensor(data['text_features'][train_mask], dtype=torch.float32)
    train_img_global = torch.tensor(data['image_features_global'][train_mask], dtype=torch.float32)
    train_img_patch = torch.tensor(data['image_features_patch'][train_mask], dtype=torch.float32)
    train_kg = torch.tensor(data['kg_features'][train_mask], dtype=torch.float32)
    train_rel_ids = torch.tensor(data['relation_ids'][train_mask], dtype=torch.long)
    train_labels = torch.tensor(data['labels_fine'][train_mask], dtype=torch.long)
    train_y_ck = torch.tensor(data['y_ck'][train_mask], dtype=torch.float32)
    train_sample_ids = torch.tensor(data['sample_ids'][train_mask], dtype=torch.long)
    
    # Val tensors
    val_text = torch.tensor(data['text_features'][val_mask], dtype=torch.float32)
    val_img_global = torch.tensor(data['image_features_global'][val_mask], dtype=torch.float32)
    val_img_patch = torch.tensor(data['image_features_patch'][val_mask], dtype=torch.float32)
    val_kg = torch.tensor(data['kg_features'][val_mask], dtype=torch.float32)
    val_rel_ids = torch.tensor(data['relation_ids'][val_mask], dtype=torch.long)
    val_labels = torch.tensor(data['labels_fine'][val_mask], dtype=torch.long)
    val_y_ck = torch.tensor(data['y_ck'][val_mask], dtype=torch.float32)
    val_sample_ids = torch.tensor(data['sample_ids'][val_mask], dtype=torch.long)
    
    print(f"Train set shape: {train_text.shape}, Val set shape: {val_text.shape}")
    
    # Compute class-weighted loss weights from train labels
    counts = np.bincount(data['labels_fine'][train_mask], minlength=num_classes)
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = len(train_labels) / (num_classes * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class counts in train: {counts.tolist()}")
    print(f"Calculated class weights: {weights.tolist()}")
    
    # Construct PyTorch DataLoaders
    train_dataset = TensorDataset(
        train_text, train_img_global, train_img_patch, train_kg,
        train_rel_ids, train_labels, train_y_ck, train_sample_ids
    )
    val_dataset = TensorDataset(
        val_text, val_img_global, val_img_patch, val_kg,
        val_rel_ids, val_labels, val_y_ck, val_sample_ids
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    num_relations = int(data['relation_ids'].max()) + 1
    model = CIKDCKBoostMoE(num_relations=num_relations, kg_dim=data['kg_features'].shape[1]).to(device)
    
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_tvcs = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    checkpoint_dir = "checkpoints/cikd"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"cikd_ckboost_moe_lambda{args.lambda_tvcs}_seed{args.seed}.pt")
    
    best_score = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    patience = 5
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        train_cls_loss_sum = 0.0
        train_base_loss_sum = 0.0
        train_tvcs_loss_sum = 0.0
        train_ck_binary_loss_sum = 0.0
        train_total_loss_sum = 0.0
        train_tvcs_count = 0
        train_count = 0
        
        for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, _ in train_loader:
            bx_text = bx_text.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_img_patch = bx_img_patch.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_label = bx_label.to(device)
            bx_y_ck = bx_y_ck.to(device)
            
            optimizer.zero_grad()
            logits_final, logits_base, logits_tvcs, c_logits, g = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
            
            # L_cls
            loss_cls = criterion_cls(logits_final, bx_label)
            
            # L_base
            loss_base = criterion_cls(logits_base, bx_label)
            
            # L_tvcs
            mask = (bx_y_ck != -1)
            if mask.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
            else:
                loss_tvcs = torch.tensor(0.0, device=device)
                
            # L_ck_binary
            target = (bx_label == 2).float()
            pred_logit = logits_final[:, 2]
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
            probs = torch.sigmoid(pred_logit)
            p_t = probs * target + (1 - probs) * (1 - target)
            focal_weight = 1 - p_t # gamma = 1.0
            alpha_factor = 0.75 * target + 0.25 * (1 - target) # alpha = 0.75
            loss_focal = alpha_factor * focal_weight * bce_loss
            loss_ck_binary = loss_focal.mean()
            
            # total loss
            loss_total = loss_cls + 0.3 * loss_base + args.lambda_tvcs * loss_tvcs + 0.2 * loss_ck_binary
            
            loss_total.backward()
            
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # accumulate losses
            train_cls_loss_sum += loss_cls.item() * len(bx_label)
            train_base_loss_sum += loss_base.item() * len(bx_label)
            train_ck_binary_loss_sum += loss_ck_binary.item() * len(bx_label)
            train_count += len(bx_label)
            if mask.sum() > 0:
                train_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                train_tvcs_count += mask.sum().item()
            train_total_loss_sum += loss_total.item() * len(bx_label)
            
        train_cls_loss = train_cls_loss_sum / train_count
        train_base_loss = train_base_loss_sum / train_count
        train_tvcs_loss = train_tvcs_loss_sum / train_tvcs_count if train_tvcs_count > 0 else 0.0
        train_ck_binary_loss = train_ck_binary_loss_sum / train_count
        train_total_loss = train_total_loss_sum / train_count
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_c_logits = []
        val_y_ck_list = []
        
        val_cls_loss_sum = 0.0
        val_base_loss_sum = 0.0
        val_tvcs_loss_sum = 0.0
        val_ck_binary_loss_sum = 0.0
        val_tvcs_count = 0
        val_count = 0
        
        with torch.no_grad():
            for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, _ in val_loader:
                bx_text = bx_text.to(device)
                bx_img_global = bx_img_global.to(device)
                bx_img_patch = bx_img_patch.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)
                bx_label = bx_label.to(device)
                bx_y_ck = bx_y_ck.to(device)
                
                logits_final, logits_base, logits_tvcs, c_logits, g = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
                
                loss_cls = criterion_cls(logits_final, bx_label)
                val_cls_loss_sum += loss_cls.item() * len(bx_label)
                
                loss_base = criterion_cls(logits_base, bx_label)
                val_base_loss_sum += loss_base.item() * len(bx_label)
                
                # L_ck_binary
                target = (bx_label == 2).float()
                pred_logit = logits_final[:, 2]
                bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
                probs = torch.sigmoid(pred_logit)
                p_t = probs * target + (1 - probs) * (1 - target)
                focal_weight = 1 - p_t # gamma = 1.0
                alpha_factor = 0.75 * target + 0.25 * (1 - target) # alpha = 0.75
                loss_focal = alpha_factor * focal_weight * bce_loss
                loss_ck_binary = loss_focal.mean()
                val_ck_binary_loss_sum += loss_ck_binary.item() * len(bx_label)
                
                val_count += len(bx_label)
                
                mask = (bx_y_ck != -1)
                if mask.sum() > 0:
                    loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
                    val_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                    val_tvcs_count += mask.sum().item()
                    
                preds = torch.argmax(logits_final, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(bx_label.cpu().numpy())
                val_c_logits.extend(c_logits.cpu().numpy())
                val_y_ck_list.extend(bx_y_ck.cpu().numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_c_logits = np.array(val_c_logits)
        val_c_probs = 1.0 / (1.0 + np.exp(-val_c_logits))
        val_y_ck_arr = np.array(val_y_ck_list)
        
        val_cls_loss = val_cls_loss_sum / val_count
        val_base_loss = val_base_loss_sum / val_count
        val_tvcs_loss = val_tvcs_loss_sum / val_tvcs_count if val_tvcs_count > 0 else 0.0
        val_ck_binary_loss = val_ck_binary_loss_sum / val_count
        val_total_loss = val_cls_loss + 0.3 * val_base_loss + args.lambda_tvcs * val_tvcs_loss + 0.2 * val_ck_binary_loss
        
        acc = accuracy_score(val_targets, val_preds)
        macro_f1 = f1_score(val_targets, val_preds, average='macro')
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
        ck_f1 = per_class_f1[2]
        sel_score = 0.5 * macro_f1 + 0.5 * ck_f1
        
        # TVCS metrics
        tvcs_mask = (val_y_ck_arr != -1)
        val_y_ck_tvcs = val_y_ck_arr[tvcs_mask]
        val_c_probs_tvcs = val_c_probs[tvcs_mask]
        
        if len(np.unique(val_y_ck_tvcs)) > 1:
            from sklearn.metrics import roc_auc_score
            tvcs_auc_ck_vs_real = roc_auc_score(val_y_ck_tvcs, val_c_probs_tvcs)
        else:
            tvcs_auc_ck_vs_real = 0.5
            
        real_mask = (val_y_ck_arr == 0)
        mean_c_real = float(np.mean(val_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
        
        ck_mask_y = (val_y_ck_arr == 1)
        mean_c_ck = float(np.mean(val_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
        
        tvcs_delta = mean_c_ck - mean_c_real
        
        # Print metrics
        print(f"cikd_ckboost_moe, epoch {epoch+1}:")
        print(f"  accuracy: {acc:.4f}")
        print(f"  macro_f1: {macro_f1:.4f}")
        print(f"  weighted_f1: {weighted_f1:.4f}")
        print(f"  ck_f1: {ck_f1:.4f}")
        print(f"  selection_score: {sel_score:.4f}")
        print(f"  tvcs_auc_ck_vs_real: {tvcs_auc_ck_vs_real:.4f}")
        print(f"  mean_c_real: {mean_c_real:.4f}")
        print(f"  mean_c_ck: {mean_c_ck:.4f}")
        print(f"  tvcs_delta: {tvcs_delta:.4f}")
        print(f"  cls_loss: {val_cls_loss:.4f}")
        print(f"  tvcs_loss: {val_tvcs_loss:.4f}")
        print(f"  total_loss: {val_total_loss:.4f}")
        
        if sel_score > best_score:
            best_score = sel_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'selection_score': best_score,
                'val_metrics': {
                    'accuracy': acc,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'ck_f1': ck_f1,
                    'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
                    'mean_c_real': mean_c_real,
                    'mean_c_ck': mean_c_ck,
                    'tvcs_delta': tvcs_delta,
                    'cls_loss': val_cls_loss,
                    'tvcs_loss': val_tvcs_loss,
                    'total_loss': val_total_loss,
                    'per_class_f1': per_class_f1.tolist()
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch} with selection score {best_score:.4f}.")
                break
                
    # Load best checkpoint for final evaluation
    print(f"Loading best checkpoint from {checkpoint_path} for final validation evaluation...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    val_preds = []
    val_targets = []
    val_c_logits = []
    val_y_ck_list = []
    val_sample_ids_list = []
    
    val_cls_loss_sum = 0.0
    val_base_loss_sum = 0.0
    val_tvcs_loss_sum = 0.0
    val_ck_binary_loss_sum = 0.0
    val_tvcs_count = 0
    val_count = 0
    
    with torch.no_grad():
        for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel, bx_label, bx_y_ck, bx_sample_id in val_loader:
            bx_text = bx_text.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_img_patch = bx_img_patch.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_label = bx_label.to(device)
            bx_y_ck = bx_y_ck.to(device)
            
            logits_final, logits_base, logits_tvcs, c_logits, g = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
            
            loss_cls = criterion_cls(logits_final, bx_label)
            val_cls_loss_sum += loss_cls.item() * len(bx_label)
            
            loss_base = criterion_cls(logits_base, bx_label)
            val_base_loss_sum += loss_base.item() * len(bx_label)
            
            # L_ck_binary
            target = (bx_label == 2).float()
            pred_logit = logits_final[:, 2]
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
            probs = torch.sigmoid(pred_logit)
            p_t = probs * target + (1 - probs) * (1 - target)
            focal_weight = 1 - p_t # gamma = 1.0
            alpha_factor = 0.75 * target + 0.25 * (1 - target) # alpha = 0.75
            loss_focal = alpha_factor * focal_weight * bce_loss
            loss_ck_binary = loss_focal.mean()
            val_ck_binary_loss_sum += loss_ck_binary.item() * len(bx_label)
            
            val_count += len(bx_label)
            
            mask = (bx_y_ck != -1)
            if mask.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logits[mask], bx_y_ck[mask])
                val_tvcs_loss_sum += loss_tvcs.item() * mask.sum().item()
                val_tvcs_count += mask.sum().item()
                
            preds = torch.argmax(logits_final, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(bx_label.cpu().numpy())
            val_c_logits.extend(c_logits.cpu().numpy())
            val_y_ck_list.extend(bx_y_ck.cpu().numpy())
            val_sample_ids_list.extend(bx_sample_id.cpu().numpy())
            
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_c_logits = np.array(val_c_logits)
    val_c_probs = 1.0 / (1.0 + np.exp(-val_c_logits))
    val_y_ck_arr = np.array(val_y_ck_list)
    val_sample_ids_arr = np.array(val_sample_ids_list)
    
    val_cls_loss = val_cls_loss_sum / val_count
    val_base_loss = val_base_loss_sum / val_count
    val_tvcs_loss = val_tvcs_loss_sum / val_tvcs_count if val_tvcs_count > 0 else 0.0
    val_ck_binary_loss = val_ck_binary_loss_sum / val_count
    val_total_loss = val_cls_loss + 0.3 * val_base_loss + args.lambda_tvcs * val_tvcs_loss + 0.2 * val_ck_binary_loss
    
    acc = accuracy_score(val_targets, val_preds)
    macro_f1 = f1_score(val_targets, val_preds, average='macro')
    weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
    per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(num_classes)))
    ck_f1 = per_class_f1[2]
    sel_score = 0.5 * macro_f1 + 0.5 * ck_f1
    
    # TVCS metrics
    tvcs_mask = (val_y_ck_arr != -1)
    val_y_ck_tvcs = val_y_ck_arr[tvcs_mask]
    val_c_probs_tvcs = val_c_probs[tvcs_mask]
    
    if len(np.unique(val_y_ck_tvcs)) > 1:
        from sklearn.metrics import roc_auc_score
        tvcs_auc_ck_vs_real = roc_auc_score(val_y_ck_tvcs, val_c_probs_tvcs)
    else:
        tvcs_auc_ck_vs_real = 0.5
        
    real_mask = (val_y_ck_arr == 0)
    mean_c_real = float(np.mean(val_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
    
    ck_mask_y = (val_y_ck_arr == 1)
    mean_c_ck = float(np.mean(val_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
    
    tvcs_delta = mean_c_ck - mean_c_real
    
    print(f"\nFinished Training. Final Best Val Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | CK-F1: {ck_f1:.4f}")
    
    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 03_cikd_metrics_val.csv
    cikd_metrics = [{
        'model': 'cikd_ckboost_moe',
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1,
        'selection_score': sel_score,
        'cls_loss': val_cls_loss,
        'total_loss': val_total_loss
    }]
    df_cikd_metrics = pd.DataFrame(cikd_metrics)
    cikd_metrics_path = os.path.join(args.out_dir, '03_cikd_metrics_val.csv')
    df_cikd_metrics.to_csv(cikd_metrics_path, index=False)
    print(f"Saved CIKD metrics to: {cikd_metrics_path}")
    
    # 2. 03_tvcs_metrics_val.csv
    tvcs_metrics = [{
        'model': 'cikd_ckboost_moe',
        'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
        'mean_c_real': mean_c_real,
        'mean_c_ck': mean_c_ck,
        'tvcs_delta': tvcs_delta,
        'tvcs_loss': val_tvcs_loss
    }]
    df_tvcs_metrics = pd.DataFrame(tvcs_metrics)
    tvcs_metrics_path = os.path.join(args.out_dir, '03_tvcs_metrics_val.csv')
    df_tvcs_metrics.to_csv(tvcs_metrics_path, index=False)
    print(f"Saved TVCS metrics to: {tvcs_metrics_path}")
    
    # 3. 03_per_class_f1_val.csv
    per_class_row = {'model': 'cikd_ckboost_moe'}
    for c in range(num_classes):
        per_class_row[f'f1_class_{c}'] = per_class_f1[c]
    df_per_class = pd.DataFrame([per_class_row])
    per_class_path = os.path.join(args.out_dir, '03_per_class_f1_val.csv')
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved per-class F1 to: {per_class_path}")
    
    # 4. 03_tvcs_scores_best_val.csv
    df_tvcs_scores = pd.DataFrame({
        'sample_id': val_sample_ids_arr,
        'y_ck': val_y_ck_arr,
        'c_logit': val_c_logits,
        'c_score': val_c_probs
    })
    tvcs_scores_path = os.path.join(args.out_dir, '03_tvcs_scores_best_val.csv')
    df_tvcs_scores.to_csv(tvcs_scores_path, index=False)
    print(f"Saved best TVCS scores to: {tvcs_scores_path}")
    
    # 5. 03_tvcs_hist_best.png
    plt.figure(figsize=(8, 6))
    real_scores = val_c_probs[val_y_ck_arr == 0]
    ck_scores = val_c_probs[val_y_ck_arr == 1]
    
    plt.hist(real_scores, bins=30, alpha=0.5, label=f'Real (y_ck=0, mean={mean_c_real:.4f})', color='green', edgecolor='k')
    plt.hist(ck_scores, bins=30, alpha=0.5, label=f'Contradictory (y_ck=1, mean={mean_c_ck:.4f})', color='red', edgecolor='k')
    
    plt.xlabel('Contradiction Score (c_score)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Contradiction Scores (Val Set, tvcs_delta={tvcs_delta:.4f})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    hist_path = os.path.join(args.out_dir, '03_tvcs_hist_best.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved TVCS scores histogram to: {hist_path}")


def main():
    parser = argparse.ArgumentParser(description="CIKD Stage C / D PyTorch Training Script")
    parser.add_argument('--stage', type=str, required=True, choices=['C', 'c', 'D', 'd'],
                        help="Training stage: 'C' (baselines) or 'D' (CIKD models).")
    parser.add_argument('--cache_dir', type=str, default='data/cache',
                        help="Path to cache directory containing feature arrays.")
    parser.add_argument('--out_dir', type=str, default='outputs/stage_c_baselines',
                        help="Path to output directory for saving evaluation CSVs.")
    parser.add_argument('--task', type=str, default='fine6',
                        help="Task identifier (only 'fine6' is supported in Stage C).")
    parser.add_argument('--subset', type=str, default='kg_complete',
                        help="Subset of cached data to use (e.g. 'kg_complete', 'full', 'tvcs_eligible').")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--model', type=str, default='cikd_light',
                        help="Stage D model architecture name (unused in Stage C).")
    parser.add_argument('--lambda_tvcs', type=float, default=0.3,
                        help="Stage D loss scaling factor for TVCS (unused in Stage C).")
                        
    args = parser.parse_args()
    
    # Normalize stage string to uppercase
    args.stage = args.stage.upper()
    
    # Support and validate --help automatically provided by argparse
    
    if args.stage == 'D':
        if args.task != 'fine6':
            raise ValueError(
                f"Unsupported task: '{args.task}'. Only 'fine6' is supported."
            )
        if args.model not in ['cikd_light', 'cikd_residual_moe', 'cikd_ckboost_moe']:
            raise ValueError(
                f"Unsupported Stage D model: '{args.model}'. Only 'cikd_light', 'cikd_residual_moe', and 'cikd_ckboost_moe' are supported."
            )
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load cached arrays for Stage D
        try:
            data = load_cached_data(args.cache_dir, args.subset, stage='D')
        except Exception as e:
            print(f"Error loading cache: {e}", file=sys.stderr)
            sys.exit(1)
            
        num_classes = 6
        if args.model == 'cikd_light':
            train_cikd_light(data, num_classes, args, device)
        elif args.model == 'cikd_residual_moe':
            train_cikd_residual_moe(data, num_classes, args, device)
        elif args.model == 'cikd_ckboost_moe':
            train_cikd_ckboost_moe(data, num_classes, args, device)
        sys.exit(0)
        
    if args.task != 'fine6':
        raise ValueError(
            f"Unsupported task: '{args.task}'. Only 'fine6' is supported."
        )
        
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load cached arrays
    try:
        data = load_cached_data(args.cache_dir, args.subset)
    except Exception as e:
        print(f"Error loading cache: {e}", file=sys.stderr)
        sys.exit(1)
        
    num_classes = 6
    labels = data['labels_fine']
    split_ids = data['split_ids']
    
    # Stage C Baselines definition
    baselines = {
        'text_only': lambda d: d['text_features'],
        'image_only': lambda d: d['image_features_global'],
        'text_image_concat': lambda d: np.concatenate([d['text_features'], d['image_features_global']], axis=1),
        'text_image_kg_concat': lambda d: np.concatenate([d['text_features'], d['image_features_global'], d['kg_features']], axis=1)
    }
    
    results_summary = []
    per_class_summary = []
    
    for model_name, feature_selector in baselines.items():
        # Set seed before training each model to ensure independent reproducibility
        set_seed(args.seed)
        
        try:
            features = feature_selector(data)
        except KeyError as e:
            print(f"Skipping model {model_name} because required features are missing in {args.subset} subset: {e}", file=sys.stderr)
            continue
            
        metrics = train_baseline(model_name, features, labels, split_ids, num_classes, args, device)
        
        results_summary.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'ck_f1': metrics['ck_f1']
        })
        
        per_class_row = {'model': model_name}
        for c in range(num_classes):
            per_class_row[f'f1_class_{c}'] = metrics['per_class_f1'][c]
        per_class_summary.append(per_class_row)
        
    # Save validation CSVs
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_metrics = pd.DataFrame(results_summary)
    metrics_path = os.path.join(args.out_dir, '02_baseline_metrics_val.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved overall validation metrics to: {metrics_path}")
    
    df_per_class = pd.DataFrame(per_class_summary)
    per_class_path = os.path.join(args.out_dir, '02_per_class_f1_val.csv')
    df_per_class.to_csv(per_class_path, index=False)
    print(f"Saved per-class validation F1 to: {per_class_path}")

if __name__ == "__main__":
    main()
