"""
Stage F4: Final Lock Evaluation.
Prepares the final test evaluation, paired bootstrapping, and reporting for CIKD++-RT.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# Import model architectures
from models.cikd_pp_rt import CIKDPPResidualTransformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SimpleMLP(nn.Module):
    """
    Simple MLP Classifier.
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

class CIKDCKBoostMoE(nn.Module):
    """
    CIKD CK-Boosted Residual Mixture of Experts (MoE) Model.
    Copied from run_stage_cd.py for self-contained execution.
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

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

def compute_metrics(true_labels, pred_labels, c_scores=None, y_ck=None):
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
            tvcs_pr_auc = average_precision_score(y_ck_tvcs, c_probs_tvcs)
        else:
            tvcs_auc_ck_vs_real = 0.5
            tvcs_pr_auc = 0.0
            
        real_mask = (y_ck == 0)
        mean_c_real = float(np.mean(c_scores[real_mask])) if real_mask.sum() > 0 else 0.0
        
        ck_mask_y = (y_ck == 1)
        mean_c_ck = float(np.mean(c_scores[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
        
        tvcs_delta = mean_c_ck - mean_c_real
        
        metrics.update({
            'tvcs_auc_ck_vs_real': tvcs_auc_ck_vs_real,
            'mean_c_real': mean_c_real,
            'mean_c_ck': mean_c_ck,
            'tvcs_delta': tvcs_delta,
            'tvcs_pr_auc': tvcs_pr_auc
        })
        
    return metrics

def run_bootstrap(true_labels, baseline_preds, model_preds, num_resamples=1000, seed=42):
    np.random.seed(seed)
    n_samples = len(true_labels)
    
    original_baseline_acc = accuracy_score(true_labels, baseline_preds)
    original_baseline_macro = f1_score(true_labels, baseline_preds, average='macro', zero_division=0)
    original_baseline_ck = f1_score(true_labels, baseline_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    original_model_acc = accuracy_score(true_labels, model_preds)
    original_model_macro = f1_score(true_labels, model_preds, average='macro', zero_division=0)
    original_model_ck = f1_score(true_labels, model_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    boot_diffs = {'accuracy': [], 'macro_f1': [], 'ck_f1': []}
    
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
            baseline_m, model_m = original_baseline_acc, original_model_acc
        elif metric == 'macro_f1':
            baseline_m, model_m = original_baseline_macro, original_model_macro
        else:
            baseline_m, model_m = original_baseline_ck, original_model_ck
            
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate CIKD++-RT on locked test split")
    parser.add_argument('--cache_dir', type=str, default='data/cache/kg_complete', help='Cache directory')
    parser.add_argument('--baseline_logits_dir', type=str, default='outputs/stage_f0_baseline_anchor', help='Baseline logits directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stage_f/cikd_pp_rt_best.pt', help='CIKD++-RT checkpoint')
    parser.add_argument('--out_dir', type=str, default='outputs/stage_f4_final_lock', help='Output directory')
    parser.add_argument('--bootstrap', type=int, default=1000, help='Bootstrap resamples count')
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
    
    # Execution / Guardrails
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry-run check without evaluating')
    parser.add_argument('--execute_final', action='store_true', help='Execute actual test evaluation')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing outputs')
    parser.add_argument("--ablation_no_c_emb", action="store_true", help="Evaluate no_c_emb ablation checkpoint correctly")
    parser.add_argument("--ablation_no_residual", action="store_true", help="Evaluate no_residual ablation checkpoint correctly")
    parser.add_argument("--ablation_global_only", action="store_true", help="Evaluate global_only ablation checkpoint correctly")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("Stage F4: Final Lock Evaluation")
    print("=" * 80)

    # Check that required model checkpoints exist
    baseline_chk_path = 'checkpoints/baselines/text_image_kg_concat_seed42.pt'
    old_cikd_chk_path = 'checkpoints/cikd/cikd_ckboost_moe_lambda0.7_seed42.pt'
    
    checkpoints = {
        'Baseline': baseline_chk_path,
        'Old CIKD': old_cikd_chk_path,
        'CIKD++-RT': args.checkpoint
    }
    
    checkpoint_status_ok = True
    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"[-] Missing checkpoint for {name} at: {path}")
            checkpoint_status_ok = False
        else:
            print(f"[+] Found checkpoint for {name} at: {path}")

    if not checkpoint_status_ok and not args.dry_run:
        print("[-] Checkpoint verification failed.")
        sys.exit(1)

    # Check test split exists in cache
    split_path = os.path.join(args.cache_dir, 'split_ids.npy')
    if not os.path.exists(split_path):
        print(f"[-] Missing split_ids.npy file in cache: {split_path}")
        sys.exit(1)
        
    split_ids = np.load(split_path)
    test_mask = (split_ids == 2)
    test_count = int(np.sum(test_mask))
    print(f"  Test samples (split_ids == 2) in cache: {test_count}")
    if test_count == 0:
        print("[-] Error: No test samples found in the cache.")
        sys.exit(1)

    # Check test baseline logits
    test_logits_path = os.path.join(args.baseline_logits_dir, 'test_logits_base.npy')
    if not os.path.exists(test_logits_path):
        if args.dry_run:
            print("[!] Warning: Baseline test logits missing. Dry-run will proceed.")
        else:
            print(f"[-] Baseline test logits missing: {test_logits_path}. Execute Stage F0 first.")
            sys.exit(1)
    else:
        print(f"[+] Found baseline test logits: {test_logits_path}")

    # Overwrite check
    planned_outputs = [
        os.path.join(args.out_dir, '04_final_main_metrics.csv'),
        os.path.join(args.out_dir, '04_final_per_class_f1.csv'),
        os.path.join(args.out_dir, '04_final_tvcs_metrics.csv'),
        os.path.join(args.out_dir, '04_bootstrap_ci_vs_tikg.csv'),
        os.path.join(args.out_dir, '04_bootstrap_ci_vs_old_cikd.csv'),
        os.path.join(args.out_dir, '04_confusion_matrix.png'),
        os.path.join(args.out_dir, '04_tvcs_hist_test.png'),
        os.path.join(args.out_dir, '04_STAGE_F_FINAL_LOCK_SUMMARY.txt')
    ]
    
    if not args.overwrite:
        for path in planned_outputs:
            if os.path.exists(path):
                print(f"[-] Output file already exists: {path}")
                print("[-] Refusing to run or overwrite without the --overwrite flag.")
                sys.exit(1)

    # Dry-run Mode
    if args.dry_run or not args.execute_final:
        print("\n--- DRY RUN STATUS ---")
        print("DRY RUN ONLY — no final test evaluation executed.")
        print("\nPlanned Comparisons:")
        print("  1. CIKD++-RT vs Baseline (text_image_kg_concat)")
        print("  2. CIKD++-RT vs Old CIKD (cikd_ckboost_moe_lambda0.7_seed42)")
        
        print("\nPlanned Output Files:")
        for path in planned_outputs:
            print(f"  {path}")

        print("\n[+] DRY RUN COMPLETED SUCCESSFULLY.")
        print("\nNext command to run for future execution:")
        print(f"python src/stage_f4_final_lock.py --cache_dir {args.cache_dir} --baseline_logits_dir {args.baseline_logits_dir} --checkpoint {args.checkpoint} --out_dir {args.out_dir} --bootstrap {args.bootstrap} --seed {args.seed} --execute_final" + (" --overwrite" if args.overwrite else ""))
        sys.exit(0)

    # Future Execution Mode (Test Evaluation)
    print("\nStarting test evaluation on final locked split...")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load cache features for test split
    print("Loading test split features...")
    text_feat = np.load(os.path.join(args.cache_dir, 'text_features.npy'))[test_mask]
    img_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))[test_mask]
    img_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))[test_mask]
    kg_feats = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))[test_mask]
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))[test_mask]
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))[test_mask]
    y_ck = np.load(os.path.join(args.cache_dir, 'y_ck.npy'))[test_mask]

    # Load baseline logits
    test_logits_base = np.load(test_logits_path)

    # Convert to Tensors
    t_text = torch.tensor(text_feat, dtype=torch.float32)
    t_img_g = torch.tensor(img_global, dtype=torch.float32)
    t_img_p = torch.tensor(img_patch, dtype=torch.float32)
    t_kg = torch.tensor(kg_feats, dtype=torch.float32)
    t_rel = torch.tensor(relation_ids, dtype=torch.long)
    t_logits = torch.tensor(test_logits_base, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on {device}...")

    # Define model dimensions
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_feats.shape[1]

    # Model 1: Baseline simple MLP
    print("Loading Baseline model...")
    concat_features = np.concatenate([text_feat, img_global, kg_feats], axis=1)
    baseline_model = SimpleMLP(input_dim=concat_features.shape[1], num_classes=6).to(device)
    load_checkpoint(baseline_model, baseline_chk_path, device)
    baseline_model.eval()

    # Model 2: Old CIKD
    print("Loading Old CIKD model...")
    old_cikd_model = CIKDCKBoostMoE(num_relations=num_relations, kg_dim=kg_dim).to(device)
    load_checkpoint(old_cikd_model, old_cikd_chk_path, device)
    old_cikd_model.eval()

    # Model 3: CIKD++-RT
    print("Loading CIKD++-RT model...")
    rt_model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=0.2
    ).to(device)
    load_checkpoint(rt_model, args.checkpoint, device)
    rt_model.eval()

    # Dataloader for modal evaluation
    dataset = TensorDataset(t_text, t_img_g, t_img_p, t_kg, t_rel, t_logits)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    baseline_preds = []
    old_cikd_preds = []
    old_cikd_c_probs = []
    rt_preds = []
    rt_c_probs = []

    with torch.no_grad():
        for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits in loader:
            bx_text = bx_text.to(device)
            bx_img_g = bx_img_g.to(device)
            bx_img_p = bx_img_p.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_logits = bx_logits.to(device)

            # Baseline Inference
            bx_concat = torch.cat([bx_text, bx_img_g, bx_kg], dim=-1)
            b_logits = baseline_model(bx_concat)
            baseline_preds.extend(torch.argmax(b_logits, dim=-1).cpu().numpy())

            # Old CIKD Inference
            o_logits, _, _, o_c_logits, _ = old_cikd_model(bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel)
            old_cikd_preds.extend(torch.argmax(o_logits, dim=-1).cpu().numpy())
            old_cikd_c_probs.extend(torch.sigmoid(o_c_logits).cpu().numpy())

            # CIKD++-RT Inference
            outputs = rt_model(bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits,
            ablation_no_c_emb=args.ablation_no_c_emb,
            ablation_no_residual=args.ablation_no_residual,
            ablation_global_only=args.ablation_global_only)
            rt_logits_out = outputs['logits_final']
            rt_c_logits_out = outputs['c_logit']
            rt_preds.extend(torch.argmax(rt_logits_out, dim=-1).cpu().numpy())
            rt_c_probs.extend(torch.sigmoid(rt_c_logits_out).cpu().numpy())

    baseline_preds = np.array(baseline_preds)
    old_cikd_preds = np.array(old_cikd_preds)
    old_cikd_c_probs = np.array(old_cikd_c_probs)
    rt_preds = np.array(rt_preds)
    rt_c_probs = np.array(rt_c_probs)

    # Compute metrics
    metrics_baseline = compute_metrics(labels_fine, baseline_preds)
    metrics_old_cikd = compute_metrics(labels_fine, old_cikd_preds, c_scores=old_cikd_c_probs, y_ck=y_ck)
    metrics_rt = compute_metrics(labels_fine, rt_preds, c_scores=rt_c_probs, y_ck=y_ck)

    # Write final_main_metrics.csv
    main_rows = []
    for name, m in [('text_image_kg_concat', metrics_baseline), ('cikd_ckboost_moe_lambda0.7', metrics_old_cikd), ('cikd_pp_rt', metrics_rt)]:
        main_rows.append({
            'model': name,
            'accuracy': m['accuracy'],
            'macro_f1': m['macro_f1'],
            'weighted_f1': m['weighted_f1'],
            'ck_f1': m['ck_f1']
        })
    df_main = pd.DataFrame(main_rows)
    df_main.to_csv(os.path.join(args.out_dir, '04_final_main_metrics.csv'), index=False)

    # Write final_per_class_f1.csv
    per_class_rows = []
    for name, m in [('text_image_kg_concat', metrics_baseline), ('cikd_ckboost_moe_lambda0.7', metrics_old_cikd), ('cikd_pp_rt', metrics_rt)]:
        row = {'model': name}
        for c in range(6):
            row[f'f1_class_{c}'] = m['per_class_f1'][c]
        per_class_rows.append(row)
    df_per_class = pd.DataFrame(per_class_rows)
    df_per_class.to_csv(os.path.join(args.out_dir, '04_final_per_class_f1.csv'), index=False)

    # Write final_tvcs_metrics.csv
    tvcs_rows = []
    for name, m in [('cikd_ckboost_moe_lambda0.7', metrics_old_cikd), ('cikd_pp_rt', metrics_rt)]:
        tvcs_rows.append({
            'model': name,
            'tvcs_auc_ck_vs_real': m['tvcs_auc_ck_vs_real'],
            'mean_c_real': m['mean_c_real'],
            'mean_c_ck': m['mean_c_ck'],
            'tvcs_delta': m['tvcs_delta'],
            'tvcs_pr_auc': m['tvcs_pr_auc']
        })
    df_tvcs = pd.DataFrame(tvcs_rows)
    df_tvcs.to_csv(os.path.join(args.out_dir, '04_final_tvcs_metrics.csv'), index=False)

    # Bootstrap Paired comparisons
    print("Running bootstrap vs Baseline...")
    df_boot_tikg = run_bootstrap(labels_fine, baseline_preds, rt_preds, num_resamples=args.bootstrap, seed=args.seed)
    df_boot_tikg.to_csv(os.path.join(args.out_dir, '04_bootstrap_ci_vs_tikg.csv'), index=False)

    print("Running bootstrap vs Old CIKD...")
    df_boot_old = run_bootstrap(labels_fine, old_cikd_preds, rt_preds, num_resamples=args.bootstrap, seed=args.seed)
    df_boot_old.to_csv(os.path.join(args.out_dir, '04_bootstrap_ci_vs_old_cikd.csv'), index=False)

    # Plot and Save Confusion Matrix for CIKD++-RT
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(labels_fine, rt_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[f"Pred_{i}" for i in range(6)],
           yticklabels=[f"True_{i}" for i in range(6)],
           title="Confusion Matrix: CIKD++-RT",
           ylabel="True label",
           xlabel="Predicted label")
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_plot_path = os.path.join(args.out_dir, '04_confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot and Save TVCS Score histogram for CIKD++-RT
    print("Generating TVCS histogram plot...")
    plt.figure(figsize=(8, 6))
    plt.hist(rt_c_probs[y_ck == 0], bins=30, alpha=0.5, label=f"Real (mean={metrics_rt['mean_c_real']:.4f})", color='green', edgecolor='k')
    plt.hist(rt_c_probs[y_ck == 1], bins=30, alpha=0.5, label=f"Contradictory (mean={metrics_rt['mean_c_ck']:.4f})", color='red', edgecolor='k')
    plt.xlabel('Contradiction Score')
    plt.ylabel('Count')
    plt.title(f"CIKD++-RT TVCS Scores (Test split, Delta={metrics_rt['tvcs_delta']:.4f})")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    hist_plot_path = os.path.join(args.out_dir, '04_tvcs_hist_test.png')
    plt.savefig(hist_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Write summary text file
    summary_path = os.path.join(args.out_dir, '04_STAGE_F_FINAL_LOCK_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("STAGE F FINAL LOCK EVALUATION SUMMARY\n")
        f.write("=====================================\n\n")
        f.write("Main Classification Metrics:\n")
        f.write(df_main.to_string(index=False) + "\n\n")
        f.write("TVCS Specialist Contradiction Detection Metrics:\n")
        f.write(df_tvcs.to_string(index=False) + "\n\n")
        f.write("Bootstrap vs T+I+KG concat Baseline:\n")
        f.write(df_boot_tikg.to_string(index=False) + "\n\n")
        f.write("Bootstrap vs Old CIKD CKBoost MoE:\n")
        f.write(df_boot_old.to_string(index=False) + "\n\n")
    print(f"[+] Saved final lock summary text file to {summary_path}")

    print("\n[+] FINAL LOCKED TEST SPLIT EVALUATION COMPLETE.")

if __name__ == "__main__":
    main()
