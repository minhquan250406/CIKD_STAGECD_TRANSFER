"""
CIKD Stage E: Final Evidence Lock.
Evaluates baseline and CIKD models on the locked test split.
Performs paired bootstrap comparison and multiseed stability summary.
"""

import os
import sys
import argparse
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

try:
    from sklearn.metrics import average_precision_score
    AV_PREC_AVAILABLE = True
except ImportError:
    AV_PREC_AVAILABLE = False


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Model Architecture Definitions (matching run_stage_cd.py exactly)
# ==============================================================================

class SimpleMLP(nn.Module):
    """
    Simple MLP Classifier.
    input_dim -> 512 -> ReLU -> Dropout(0.2) -> 256 -> ReLU -> Dropout(0.2) -> num_classes
    """
    def __init__(self, input_dim, num_classes=6):
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


class CIKDLight(nn.Module):
    """
    CIKD-Light Model Architecture.
    """
    def __init__(self, num_relations, kg_dim=100):
        super().__init__()
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
        
        # Classifier
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
        
        cls_input = torch.cat([text_feats, img_global, z_k_cls, z_v, c_emb], dim=-1)
        logits = self.classifier(cls_input)
        
        return logits, c_logit


class CIKDResidualMoE(nn.Module):
    """
    CIKD Residual Mixture of Experts (MoE) Model.
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
        
        residual = logits_tvcs - logits_base
        logits_final = logits_base + g * residual
        
        return logits_final, logits_base, logits_tvcs, c_logit, g


class CIKDCKBoostMoE(nn.Module):
    """
    CIKD CK-Boosted Residual Mixture of Experts (MoE) Model.
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

# ==============================================================================
# Helper Functions for Evaluation and Calculations
# ==============================================================================

def load_checkpoint(model, path, device):
    """Load state dict from a saved checkpoint path."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)


def load_cached_data(cache_dir, subset):
    """Load cached features and target arrays."""
    subset_dir = os.path.join(cache_dir, subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"Subset directory does not exist: {subset_dir}")
        
    required_files = [
        'text_features.npy',
        'image_features_global.npy',
        'image_features_patch.npy',
        'kg_features.npy',
        'relation_ids.npy',
        'labels_fine.npy',
        'y_ck.npy',
        'split_ids.npy',
        'sample_ids.npy'
    ]
    
    for f in required_files:
        path = os.path.join(subset_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required cached file not found: {path}")
            
    print(f"Loading cached arrays from: {subset_dir}")
    data = {
        'text_features': np.load(os.path.join(subset_dir, 'text_features.npy'), mmap_mode='r'),
        'image_features_global': np.load(os.path.join(subset_dir, 'image_features_global.npy'), mmap_mode='r'),
        'image_features_patch': np.load(os.path.join(subset_dir, 'image_features_patch.npy'), mmap_mode='r'),
        'kg_features': np.load(os.path.join(subset_dir, 'kg_features.npy'), mmap_mode='r'),
        'relation_ids': np.load(os.path.join(subset_dir, 'relation_ids.npy'), mmap_mode='r'),
        'labels_fine': np.load(os.path.join(subset_dir, 'labels_fine.npy'), mmap_mode='r'),
        'y_ck': np.load(os.path.join(subset_dir, 'y_ck.npy'), mmap_mode='r'),
        'split_ids': np.load(os.path.join(subset_dir, 'split_ids.npy'), mmap_mode='r'),
        'sample_ids': np.load(os.path.join(subset_dir, 'sample_ids.npy'), mmap_mode='r')
    }
    return data


def compute_metrics(true_labels, pred_labels, c_scores=None, y_ck=None):
    """Compute overall, per-class F1, and TVCS metrics."""
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    per_class_f1 = f1_score(true_labels, pred_labels, average=None, labels=list(range(6)), zero_division=0)
    ck_f1 = per_class_f1[2]
    
    metrics = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1,
        'per_class_f1': per_class_f1
    }
    
    if c_scores is not None and y_ck is not None:
        tvcs_mask = (y_ck != -1)
        y_ck_tvcs = y_ck[tvcs_mask]
        c_probs_tvcs = c_scores[tvcs_mask]
        
        if len(np.unique(y_ck_tvcs)) > 1:
            tvcs_auc_ck_vs_real = roc_auc_score(y_ck_tvcs, c_probs_tvcs)
        else:
            tvcs_auc_ck_vs_real = 0.5
            
        real_mask = (y_ck == 0)
        mean_c_real = float(np.mean(c_scores[real_mask])) if real_mask.sum() > 0 else 0.0
        
        ck_mask_y = (y_ck == 1)
        mean_c_ck = float(np.mean(c_scores[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
        
        tvcs_delta = mean_c_ck - mean_c_real
        
        metrics.update({
            'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
            'mean_c_real': mean_c_real,
            'mean_c_ck': mean_c_ck,
            'tvcs_delta': tvcs_delta
        })
        
        if AV_PREC_AVAILABLE:
            if len(np.unique(y_ck_tvcs)) > 1:
                tvcs_pr_auc = average_precision_score(y_ck_tvcs, c_probs_tvcs)
            else:
                tvcs_pr_auc = 0.0
            metrics['tvcs_pr_auc'] = tvcs_pr_auc
        else:
            metrics['tvcs_pr_auc'] = np.nan
            
    return metrics


def run_bootstrap(true_labels, baseline_preds, model_preds, num_resamples=1000, seed=42):
    """Run paired bootstrap comparisons between main model and baseline model."""
    np.random.seed(seed)
    n_samples = len(true_labels)
    
    # Calculate original metrics
    original_baseline_acc = accuracy_score(true_labels, baseline_preds)
    original_baseline_macro = f1_score(true_labels, baseline_preds, average='macro', zero_division=0)
    original_baseline_ck = f1_score(true_labels, baseline_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    original_model_acc = accuracy_score(true_labels, model_preds)
    original_model_macro = f1_score(true_labels, model_preds, average='macro', zero_division=0)
    original_model_ck = f1_score(true_labels, model_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    boot_diffs = {
        'accuracy': [],
        'macro_f1': [],
        'ck_f1': []
    }
    
    for _ in range(num_resamples):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_true = true_labels[indices]
        boot_baseline = baseline_preds[indices]
        boot_model = model_preds[indices]
        
        b_acc = accuracy_score(boot_true, boot_baseline)
        b_macro = f1_score(boot_true, boot_baseline, average='macro', zero_division=0)
        b_ck = f1_score(boot_true, boot_baseline, average=None, labels=list(range(6)), zero_division=0)[2]
        
        m_acc = accuracy_score(boot_true, boot_model)
        m_macro = f1_score(boot_true, boot_model, average='macro', zero_division=0)
        m_ck = f1_score(boot_true, boot_model, average=None, labels=list(range(6)), zero_division=0)[2]
        
        boot_diffs['accuracy'].append(m_acc - b_acc)
        boot_diffs['macro_f1'].append(m_macro - b_macro)
        boot_diffs['ck_f1'].append(m_ck - b_ck)
        
    bootstrap_results = []
    
    for metric in ['macro_f1', 'ck_f1', 'accuracy']:
        diffs = np.array(boot_diffs[metric])
        mean_diff = np.mean(diffs)
        ci95_low = np.percentile(diffs, 2.5)
        ci95_high = np.percentile(diffs, 97.5)
        improvement_prob = np.mean(diffs > 0)
        
        if metric == 'accuracy':
            baseline_m = original_baseline_acc
            model_m = original_model_acc
        elif metric == 'macro_f1':
            baseline_m = original_baseline_macro
            model_m = original_model_macro
        else:
            baseline_m = original_baseline_ck
            model_m = original_model_ck
            
        bootstrap_results.append({
            'metric': metric,
            'baseline_metric': baseline_m,
            'model_metric': model_m,
            'mean_diff': mean_diff,
            'ci95_low': ci95_low,
            'ci95_high': ci95_high,
            'improvement_probability': improvement_prob
        })
        
    return pd.DataFrame(bootstrap_results)


# ==============================================================================
# Main Script Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CIKD Stage E: Final Evidence Lock")
    parser.add_argument('--cache_dir', type=str, default='data/cache', help='Path to cached arrays directory')
    parser.add_argument('--subset', type=str, default='kg_complete', help='Cached subset name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--out_dir', type=str, default='outputs/stage_e_final_lock', help='Output directory')
    parser.add_argument('--bootstrap', type=int, default=1000, help='Bootstrap resamples count')
    parser.add_argument('--seed', type=int, default=42, help='Primary reproducibility seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create outputs directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load cached arrays
    data = load_cached_data(args.cache_dir, args.subset)
    
    # Extract test split only (split_ids == 2)
    split_ids = data['split_ids']
    test_mask = (split_ids == 2)
    
    test_sample_count = int(np.sum(test_mask))
    if test_sample_count == 0:
        raise ValueError("No test samples found in the cache (split_ids == 2).")
    print(f"Number of test samples (split_ids == 2): {test_sample_count}")
    
    test_labels = data['labels_fine'][test_mask]
    test_y_ck = data['y_ck'][test_mask]
    test_sample_ids = data['sample_ids'][test_mask]
    
    # Extract parameters for CIKD architectures
    num_relations = int(data['relation_ids'].max()) + 1
    kg_dim = data['kg_features'].shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define models to evaluate
    models_to_eval = [
        # Stage C Baselines
        {
            'name': 'text_only',
            'type': 'baseline',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'baselines', 'text_only_seed42.pt'),
            'required': False,
            'features_fn': lambda d: d['text_features']
        },
        {
            'name': 'image_only',
            'type': 'baseline',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'baselines', 'image_only_seed42.pt'),
            'required': False,
            'features_fn': lambda d: d['image_features_global']
        },
        {
            'name': 'text_image_concat',
            'type': 'baseline',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'baselines', 'text_image_concat_seed42.pt'),
            'required': False,
            'features_fn': lambda d: np.concatenate([d['text_features'], d['image_features_global']], axis=1)
        },
        {
            'name': 'text_image_kg_concat',
            'type': 'baseline',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'baselines', 'text_image_kg_concat_seed42.pt'),
            'required': True,
            'features_fn': lambda d: np.concatenate([d['text_features'], d['image_features_global'], d['kg_features']], axis=1)
        },
        # Stage D CIKD Candidates
        {
            'name': 'cikd_light_lambda0.5_seed42',
            'type': 'cikd_light',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'cikd', 'cikd_light_lambda0.5_seed42.pt'),
            'required': False
        },
        {
            'name': 'cikd_residual_moe_lambda0.5_seed42',
            'type': 'cikd_residual_moe',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'cikd', 'cikd_residual_moe_lambda0.5_seed42.pt'),
            'required': False
        },
        {
            'name': 'cikd_residual_moe_lambda0.7_seed42',
            'type': 'cikd_residual_moe',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'cikd', 'cikd_residual_moe_lambda0.7_seed42.pt'),
            'required': False
        },
        {
            'name': 'cikd_ckboost_moe_lambda0.7_seed42',
            'type': 'cikd_ckboost_moe',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'cikd', 'cikd_ckboost_moe_lambda0.7_seed42.pt'),
            'required': True
        },
        {
            'name': 'cikd_ckboost_moe_lambda0.7_seed43',
            'type': 'cikd_ckboost_moe',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'cikd', 'cikd_ckboost_moe_lambda0.7_seed43.pt'),
            'required': False
        },
        {
            'name': 'cikd_ckboost_moe_lambda0.7_seed44',
            'type': 'cikd_ckboost_moe',
            'checkpoint_path': os.path.join(args.checkpoint_dir, 'cikd', 'cikd_ckboost_moe_lambda0.7_seed44.pt'),
            'required': False
        }
    ]
    
    evaluated_models_metrics = {}
    evaluated_models_preds = {}
    
    main_metrics_rows = []
    per_class_f1_rows = []
    tvcs_metrics_rows = []
    
    predictions_df_dict = {
        'sample_id': test_sample_ids,
        'true_label': test_labels
    }
    
    for m_info in models_to_eval:
        m_name = m_info['name']
        m_type = m_info['type']
        ckpt_path = m_info['checkpoint_path']
        required = m_info['required']
        
        if not os.path.exists(ckpt_path):
            if required:
                raise FileNotFoundError(f"Required checkpoint file is missing: {ckpt_path}")
            else:
                warnings.warn(f"Optional checkpoint is missing, skipping: {ckpt_path}")
                continue
                
        print(f"Evaluating {m_name} from {ckpt_path}...")
        
        if m_type == 'baseline':
            full_feats = m_info['features_fn'](data)
            test_feats = full_feats[test_mask]
            
            model = SimpleMLP(input_dim=test_feats.shape[1], num_classes=6).to(device)
            load_checkpoint(model, ckpt_path, device)
            model.eval()
            
            ds = TensorDataset(torch.tensor(test_feats, dtype=torch.float32))
            loader = DataLoader(ds, batch_size=128, shuffle=False)
            
            all_preds = []
            with torch.no_grad():
                for (bx,) in loader:
                    bx = bx.to(device)
                    logits = model(bx)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
            
            preds_np = np.array(all_preds)
            c_scores_np = None
            
        else:
            if m_type == 'cikd_light':
                model = CIKDLight(num_relations=num_relations, kg_dim=kg_dim).to(device)
            elif m_type == 'cikd_residual_moe':
                model = CIKDResidualMoE(num_relations=num_relations, kg_dim=kg_dim).to(device)
            elif m_type == 'cikd_ckboost_moe':
                model = CIKDCKBoostMoE(num_relations=num_relations, kg_dim=kg_dim).to(device)
            else:
                raise ValueError(f"Unknown model type: {m_type}")
                
            load_checkpoint(model, ckpt_path, device)
            model.eval()
            
            test_text = torch.tensor(data['text_features'][test_mask], dtype=torch.float32)
            test_img_global = torch.tensor(data['image_features_global'][test_mask], dtype=torch.float32)
            test_img_patch = torch.tensor(data['image_features_patch'][test_mask], dtype=torch.float32)
            test_kg = torch.tensor(data['kg_features'][test_mask], dtype=torch.float32)
            test_rel_ids = torch.tensor(data['relation_ids'][test_mask], dtype=torch.long)
            
            ds = TensorDataset(test_text, test_img_global, test_img_patch, test_kg, test_rel_ids)
            loader = DataLoader(ds, batch_size=128, shuffle=False)
            
            all_preds = []
            all_c_scores = []
            
            with torch.no_grad():
                for bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel in loader:
                    bx_text = bx_text.to(device)
                    bx_img_global = bx_img_global.to(device)
                    bx_img_patch = bx_img_patch.to(device)
                    bx_kg = bx_kg.to(device)
                    bx_rel = bx_rel.to(device)
                    
                    if m_type == 'cikd_light':
                        logits, c_logits = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
                    else:
                        logits, _, _, c_logits, _ = model(bx_text, bx_img_global, bx_img_patch, bx_kg, bx_rel)
                        
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    c_probs = torch.sigmoid(c_logits).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_c_scores.extend(c_probs)
                    
            preds_np = np.array(all_preds)
            c_scores_np = np.array(all_c_scores)
            
        metrics = compute_metrics(test_labels, preds_np, c_scores=c_scores_np, y_ck=test_y_ck)
        
        evaluated_models_metrics[m_name] = metrics
        evaluated_models_preds[m_name] = preds_np
        
        predictions_df_dict[f'{m_name}_pred'] = preds_np
        if c_scores_np is not None:
            predictions_df_dict[f'{m_name}_c_score'] = c_scores_np
            
        main_metrics_rows.append({
            'model': m_name,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'ck_f1': metrics['ck_f1'],
            'num_samples': test_sample_count
        })
        
        per_class_row = {'model': m_name}
        for c in range(6):
            per_class_row[f'f1_class_{c}'] = metrics['per_class_f1'][c]
        per_class_f1_rows.append(per_class_row)
        
        if c_scores_np is not None:
            tvcs_metrics_rows.append({
                'model': m_name,
                'tvcs_auc_ck_vs_real': metrics['tvcs_auc_ck_vs_real'],
                'mean_c_real': metrics['mean_c_real'],
                'mean_c_ck': metrics['mean_c_ck'],
                'tvcs_delta': metrics['tvcs_delta'],
                'tvcs_pr_auc': metrics['tvcs_pr_auc']
            })
            
        if m_name in ['text_image_kg_concat', 'cikd_ckboost_moe_lambda0.7_seed42']:
            cm = confusion_matrix(test_labels, preds_np, labels=list(range(6)))
            df_cm = pd.DataFrame(
                cm,
                index=[f"true_class_{i}" for i in range(6)],
                columns=[f"pred_class_{i}" for i in range(6)]
            )
            
            if m_name == 'text_image_kg_concat':
                cm_fn = '04_confusion_matrix_text_image_kg_concat.csv'
            else:
                cm_fn = '04_confusion_matrix_cikd_ckboost_moe_lam07_seed42.csv'
                
            cm_path = os.path.join(args.out_dir, cm_fn)
            df_cm.to_csv(cm_path, index=True)
            print(f"Saved confusion matrix for {m_name} to {cm_path}")
            
    # Save CSV outputs
    df_main_metrics = pd.DataFrame(main_metrics_rows)
    df_main_metrics.to_csv(os.path.join(args.out_dir, '04_final_main_metrics.csv'), index=False)
    
    df_per_class = pd.DataFrame(per_class_f1_rows)
    df_per_class.to_csv(os.path.join(args.out_dir, '04_final_per_class_f1.csv'), index=False)
    
    df_tvcs = pd.DataFrame(tvcs_metrics_rows)
    df_tvcs.to_csv(os.path.join(args.out_dir, '04_final_tvcs_metrics.csv'), index=False)
    
    df_predictions = pd.DataFrame(predictions_df_dict)
    df_predictions.to_csv(os.path.join(args.out_dir, '04_final_predictions.csv'), index=False)
    
    print("Saved all standard evaluation CSVs.")
    
    # --------------------------------------------------------------------------
    # Paired Bootstrap Analysis
    # --------------------------------------------------------------------------
    baseline_bootstrap_name = 'text_image_kg_concat'
    model_bootstrap_name = 'cikd_ckboost_moe_lambda0.7_seed42'
    
    print(f"Running paired bootstrap comparison ({args.bootstrap} resamples): {model_bootstrap_name} vs {baseline_bootstrap_name}...")
    df_bootstrap = run_bootstrap(
        test_labels,
        evaluated_models_preds[baseline_bootstrap_name],
        evaluated_models_preds[model_bootstrap_name],
        num_resamples=args.bootstrap,
        seed=args.seed
    )
    bootstrap_path = os.path.join(args.out_dir, '04_bootstrap_ci.csv')
    df_bootstrap.to_csv(bootstrap_path, index=False)
    print(f"Saved bootstrap CI to {bootstrap_path}")
    
    # --------------------------------------------------------------------------
    # Multiseed Stability Summary
    # --------------------------------------------------------------------------
    ckboost_results = []
    for seed in [42, 43, 44]:
        name_with_seed = f'cikd_ckboost_moe_lambda0.7_seed{seed}'
        if name_with_seed in evaluated_models_metrics:
            m_metrics = evaluated_models_metrics[name_with_seed]
            ckboost_results.append({
                'seed': str(seed),
                'accuracy': m_metrics['accuracy'],
                'macro_f1': m_metrics['macro_f1'],
                'weighted_f1': m_metrics['weighted_f1'],
                'ck_f1': m_metrics['ck_f1'],
                'tvcs_auc_ck_vs_real': m_metrics.get('tvcs_auc_ck_vs_real', np.nan),
                'tvcs_delta': m_metrics.get('tvcs_delta', np.nan)
            })
            
    if ckboost_results:
        df_multiseed = pd.DataFrame(ckboost_results)
        numeric_cols = ['accuracy', 'macro_f1', 'weighted_f1', 'ck_f1', 'tvcs_auc_ck_vs_real', 'tvcs_delta']
        
        mean_row = {'seed': 'mean'}
        std_row = {'seed': 'std'}
        
        for col in numeric_cols:
            mean_row[col] = df_multiseed[col].mean()
            std_row[col] = df_multiseed[col].std(ddof=1) if len(df_multiseed) > 1 else 0.0
            
        df_multiseed = pd.concat([df_multiseed, pd.DataFrame([mean_row, std_row])], ignore_index=True)
        multiseed_path = os.path.join(args.out_dir, '04_ckboost_multiseed_test_summary.csv')
        df_multiseed.to_csv(multiseed_path, index=False)
        print(f"Saved multiseed summary to {multiseed_path}")
        
    # --------------------------------------------------------------------------
    # Text Summary Generation
    # --------------------------------------------------------------------------
    b_metrics = evaluated_models_metrics[baseline_bootstrap_name]
    m_metrics = evaluated_models_metrics[model_bootstrap_name]
    
    row_macro = df_bootstrap[df_bootstrap['metric'] == 'macro_f1'].iloc[0]
    row_ck = df_bootstrap[df_bootstrap['metric'] == 'ck_f1'].iloc[0]
    row_acc = df_bootstrap[df_bootstrap['metric'] == 'accuracy'].iloc[0]
    
    summary_text = f"""================================================================================
Stage E Final Evidence Lock Summary
================================================================================
Test sample count: {test_sample_count}

Baseline ({baseline_bootstrap_name}) Metrics:
- Accuracy: {b_metrics['accuracy']:.4f}
- Macro-F1: {b_metrics['macro_f1']:.4f}
- Weighted-F1: {b_metrics['weighted_f1']:.4f}
- CK-F1: {b_metrics['ck_f1']:.4f}

Main Model ({model_bootstrap_name}) Metrics:
- Accuracy: {m_metrics['accuracy']:.4f}
- Macro-F1: {m_metrics['macro_f1']:.4f}
- Weighted-F1: {m_metrics['weighted_f1']:.4f}
- CK-F1: {m_metrics['ck_f1']:.4f}
- TVCS AUC (CK vs Real): {m_metrics.get('tvcs_auc_ck_vs_real', np.nan):.4f}
- TVCS Delta (mean_c_ck - mean_c_real): {m_metrics.get('tvcs_delta', np.nan):.4f}

Bootstrap Results (Main vs Baseline, {args.bootstrap} resamples):
- Macro-F1 Difference: Mean Diff={row_macro['mean_diff']:.4f}, 95% CI=[{row_macro['ci95_low']:.4f}, {row_macro['ci95_high']:.4f}], Prob(Improvement)={row_macro['improvement_probability']:.4%}
- CK-F1 Difference: Mean Diff={row_ck['mean_diff']:.4f}, 95% CI=[{row_ck['ci95_low']:.4f}, {row_ck['ci95_high']:.4f}], Prob(Improvement)={row_ck['improvement_probability']:.4%}
- Accuracy Difference: Mean Diff={row_acc['mean_diff']:.4f}, 95% CI=[{row_acc['ci95_low']:.4f}, {row_acc['ci95_high']:.4f}], Prob(Improvement)={row_acc['improvement_probability']:.4%}

WARNING: No model selection was performed on the test split.
The main model remains cikd_ckboost_moe_lambda0.7_seed42, selected solely based on validation candidate lock.
================================================================================
"""
    summary_path = os.path.join(args.out_dir, '04_STAGE_E_FINAL_LOCK_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
        
    print(f"Saved text summary to {summary_path}")
    print("\nStage E Final Evidence Lock evaluation completed successfully.")


if __name__ == '__main__':
    main()
