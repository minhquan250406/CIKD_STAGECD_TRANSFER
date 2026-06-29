"""
Stage I-BC: Multitask CIKD++ Training and Validation-Only Runner.
Loads config JSON, pre-calculated features, and pre-trained checkpoints (TVCS Specialist and F4 Backbone),
trains the multitask residual architecture on the train split, validates on the validation split,
and generates validation logs and summaries.
"""

import os
import sys
import json
import argparse
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Ensure src directory is in path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stage_i_bc_multitask_balanced_model import StageIBCMultitaskCIKDPP
from stage_i_bc_losses import compute_multitask_total_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-BC Multitask Balanced Training.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON config file.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override maximum epochs.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early stopping patience.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override weight decay.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override reproducibility seed.")
    parser.add_argument("--no_test_eval", action="store_true",
                        help="Assert that test split evaluation is disabled.")
    return parser.parse_args()

def regenerate_global_reports(project_root):
    """
    Reads the config CSV and all epoch logs to dynamically regenerate
    I_BC_TRAINING_SUMMARY.txt and I_BC_FINAL_DECISION.txt.
    """
    out_dir = os.path.join(project_root, "outputs", "stage_i_macro_micro_improvement")
    report_csv = os.path.join(out_dir, "I_BC_VAL_METRICS_ALL_CONFIGS.csv")
    summary_txt = os.path.join(out_dir, "I_BC_TRAINING_SUMMARY.txt")
    decision_txt = os.path.join(out_dir, "I_BC_FINAL_DECISION.txt")
    
    if not os.path.exists(report_csv):
        return
        
    df_configs = pd.read_csv(report_csv)
    
    # 1. Regenerate I_BC_TRAINING_SUMMARY.txt
    with open(summary_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE I-BC: MULTITASK CLASS-BALANCED TRAINING SUMMARY\n")
        f.write("========================================================================\n\n")
        
        f.write("SUMMARY OF BEST EPOCH VAL PERFORMANCE ACROSS CONFIGURATIONS:\n")
        f.write("-----------------------------------------------------------\n")
        f.write(df_configs.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED EPOCH-BY-EPOCH TRAINING LOGS:\n")
        f.write("--------------------------------------\n")
        epoch_log_files = glob.glob(os.path.join(out_dir, "I_BC_EPOCH_LOG_*.csv"))
        epoch_log_files.sort()
        
        for lf in epoch_log_files:
            cfg_name = os.path.basename(lf).replace("I_BC_EPOCH_LOG_", "").replace(".csv", "").upper()
            f.write(f"\nConfiguration: {cfg_name}\n")
            f.write("=" * (len(cfg_name) + 15) + "\n")
            df_log = pd.read_csv(lf)
            f.write(df_log.to_string(index=False))
            f.write("\n")
            
    # 2. Regenerate I_BC_FINAL_DECISION.txt
    if len(df_configs) > 0:
        best_idx = df_configs["val_selection_score"].idxmax()
        best_row = df_configs.iloc[best_idx]
        
        with open(decision_txt, 'w') as f:
            f.write("========================================================================\n")
            f.write("STAGE I-BC: FINAL MODEL SELECTION DECISION\n")
            f.write("========================================================================\n\n")
            f.write("Based on the validation selection score: 0.45 * Macro-F1 + 0.35 * CK-F1 + 0.20 * TVCS_AUC\n")
            f.write(f"The best performing configuration is: {best_row['config_name']}\n\n")
            f.write("Performance Details at Best Epoch:\n")
            f.write("---------------------------------\n")
            f.write(f"Best Epoch:                 {best_row['epoch']}\n")
            f.write(f"Validation Selection Score:  {best_row['val_selection_score']:.6f}\n")
            f.write(f"Validation Accuracy:        {best_row['val_accuracy']:.6f}\n")
            f.write(f"Validation Macro-F1:       {best_row['val_macro_f1']:.6f}\n")
            f.write(f"Validation Micro-F1:       {best_row['val_micro_f1']:.6f}\n")
            f.write(f"Validation CK-F1:          {best_row['val_ck_f1']:.6f}\n")
            f.write(f"Validation TVCS AUC:       {best_row['val_tvcs_auc']:.6f}\n")
            f.write(f"Validation TVCS PR-AUC:    {best_row['val_tvcs_pr_auc']:.6f}\n")
            f.write(f"Validation TVCS Delta:     {best_row['val_tvcs_delta']:.6f}\n\n")
            f.write("Status: Validated, test split evaluation omitted per guardrail requirements.\n")

def main():
    # 1. Parse arguments and check startup assertion
    args = parse_args()
    assert args.no_test_eval, "Error: --no_test_eval flag must be present at startup to prevent accidental test split leakage."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running on device: {device}")
    
    # 2. Load and parse configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, 'r') as f:
        config_data = json.load(f)
        
    config_name = config_data.get("config_name", os.path.basename(args.config))
    print(f"[+] Loaded config: {config_name}")
    
    # Resolve hyperparams with optional cmd overrides
    epochs = args.max_epochs if args.max_epochs is not None else config_data.get("epochs", 20)
    patience = args.patience if args.patience is not None else config_data.get("patience", 5)
    batch_size = args.batch_size if args.batch_size is not None else config_data.get("batch_size", 64)
    lr = args.lr if args.lr is not None else config_data.get("lr", 0.0001)
    weight_decay = args.weight_decay if args.weight_decay is not None else config_data.get("weight_decay", 0.01)
    seed = args.seed if args.seed is not None else config_data.get("seed", 42)
    
    set_seed(seed)
    print(f"    Parameters: epochs={epochs}, patience={patience}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, seed={seed}")
    
    # 3. Verify paths & directory structures
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement")
    ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_i")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Map config to checkpoint filename
    config_basename = os.path.basename(args.config).lower()
    if config_basename == "i_bc_a.json":
        ckpt_name = "best_i_bc_a.pt"
    elif config_basename == "i_bc_b.json":
        ckpt_name = "best_i_bc_b.pt"
    elif config_basename == "i_bc_c.json":
        ckpt_name = "best_i_bc_c.pt"
    elif config_basename == "i_bc_d.json":
        ckpt_name = "best_i_bc_d.pt"
    else:
        ckpt_name = f"best_{os.path.splitext(config_basename)[0]}.pt"
    
    # 4. Load cached features from kg_complete
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    print(f"[+] Loading cached features from: {cache_dir}")
    split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
    relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
    kg_features = np.load(os.path.join(cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
    y_ck = np.load(os.path.join(cache_dir, 'y_ck.npy'))
    text_features = np.load(os.path.join(cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))
    
    # Load baseline logits from stage_f0_baseline_anchor
    baseline_dir = os.path.join(args.project_root, "outputs", "stage_f0_baseline_anchor")
    print(f"[+] Loading baseline logits from: {baseline_dir}")
    tr_logits_base = np.load(os.path.join(baseline_dir, 'train_logits_base.npy'))
    val_logits_base = np.load(os.path.join(baseline_dir, 'val_logits_base.npy'))
    
    # 5. Extract Train and Val masks (Strictly omit test split_ids == 2)
    train_mask = (split_ids == 0)
    val_mask = (split_ids == 1)
    
    assert not np.any(split_ids[train_mask] == 2), "Test split leaked into train subset!"
    assert not np.any(split_ids[val_mask] == 2), "Test split leaked into validation subset!"
    
    # Check shapes
    assert len(tr_logits_base) == np.sum(train_mask), "Train logits shape mismatch!"
    assert len(val_logits_base) == np.sum(val_mask), "Val logits shape mismatch!"
    
    # Convert datasets to torch Tensors
    tr_text = torch.tensor(text_features[train_mask], dtype=torch.float32)
    tr_img_g = torch.tensor(image_features_global[train_mask], dtype=torch.float32)
    tr_img_p = torch.tensor(image_features_patch[train_mask], dtype=torch.float32)
    tr_kg = torch.tensor(kg_features[train_mask], dtype=torch.float32)
    tr_rel = torch.tensor(relation_ids[train_mask], dtype=torch.long)
    tr_lbl = torch.tensor(labels_fine[train_mask], dtype=torch.long)
    tr_y_ck = torch.tensor(y_ck[train_mask], dtype=torch.float32)
    tr_logits = torch.tensor(tr_logits_base, dtype=torch.float32)
    
    val_text = torch.tensor(text_features[val_mask], dtype=torch.float32)
    val_img_g = torch.tensor(image_features_global[val_mask], dtype=torch.float32)
    val_img_p = torch.tensor(image_features_patch[val_mask], dtype=torch.float32)
    val_kg = torch.tensor(kg_features[val_mask], dtype=torch.float32)
    val_rel = torch.tensor(relation_ids[val_mask], dtype=torch.long)
    val_lbl = torch.tensor(labels_fine[val_mask], dtype=torch.long)
    val_y_ck = torch.tensor(y_ck[val_mask], dtype=torch.float32)
    val_logits = torch.tensor(val_logits_base, dtype=torch.float32)
    
    train_ds = TensorDataset(tr_text, tr_img_g, tr_img_p, tr_kg, tr_rel, tr_lbl, tr_y_ck, tr_logits)
    val_ds = TensorDataset(val_text, val_img_g, val_img_p, val_kg, val_rel, val_lbl, val_y_ck, val_logits)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 6. Instantiate model
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_features.shape[1]
    
    model = StageIBCMultitaskCIKDPP(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=config_data.get('dropout', 0.2),
        alpha_init=0.2,
        alpha_max=config_data['alpha_max']
    ).to(device)
    
    # 7. Load Checkpoints
    # Load TVCS specialist
    tvcs_ckpt_path = os.path.join(args.project_root, "checkpoints", "stage_f", "tvcs_specialist_seed42_padded_for_f2.pt")
    if os.path.exists(tvcs_ckpt_path):
        print(f"[+] Loading TVCS Specialist checkpoint: {tvcs_ckpt_path}")
        tvcs_ckpt = torch.load(tvcs_ckpt_path, map_location=device, weights_only=False)
        tvcs_state = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
        model.tvcs_specialist.load_state_dict(tvcs_state)
    else:
        raise FileNotFoundError(f"TVCS Specialist checkpoint missing at: {tvcs_ckpt_path}")
        
    # Load F4 backbone
    f4_ckpt_path = os.path.join(args.project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
    if os.path.exists(f4_ckpt_path):
        print(f"[+] Loading F4 Backbone checkpoint: {f4_ckpt_path}")
        f4_ckpt = torch.load(f4_ckpt_path, map_location=device, weights_only=False)
        f4_state = f4_ckpt.get('model_state_dict', f4_ckpt)
        clean_state = {}
        for k, v in f4_state.items():
            if k.startswith('base_model.'):
                clean_state[k.replace('base_model.', '')] = v
            else:
                clean_state[k] = v
        # Load with strict=False to leave binary_head and ck_real_head randomly initialized
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        print(f"    Backbone weights loaded. Missing keys ( scratch initialized ): {missing}")
    else:
        raise FileNotFoundError(f"F4 Backbone checkpoint missing at: {f4_ckpt_path}")
        
    # 8. Freeze TVCS Specialist parameters
    for param in model.tvcs_specialist.parameters():
        param.requires_grad = False
    print("[+] Frozen TVCS Specialist parameters.")
    
    # 9. Compute class priors and weights from train split labels for logit adjustment
    train_labels_np = labels_fine[train_mask]
    class_counts = np.bincount(train_labels_np, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_labels_np) / (6.0 * class_counts)
    
    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    # 10. Training Loop
    epoch_logs = []
    best_score = -1.0
    best_epoch_num = -1
    patience_counter = 0
    
    print("\n[+] Starting training loop...")
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = {
            "loss_total": 0.0,
            "loss_6way": 0.0,
            "loss_binary": 0.0,
            "loss_ck_real": 0.0,
            "loss_kl": 0.0,
            "loss_residual": 0.0
        }
        total_samples = 0
        
        for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_y_ck, bx_logits in train_loader:
            bx_text = bx_text.to(device)
            bx_img_g = bx_img_g.to(device)
            bx_img_p = bx_img_p.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_lbl = bx_lbl.to(device)
            bx_y_ck = bx_y_ck.to(device)
            bx_logits = bx_logits.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                text_features=bx_text,
                image_global_features=bx_img_g,
                image_patch_features=bx_img_p,
                kg_features=bx_kg,
                relation_ids=bx_rel,
                baseline_logits=bx_logits,
                ablation_no_c_emb=True
            )
            
            loss_outputs = compute_multitask_total_loss(
                logits_final=outputs['logits_final'],
                logits_delta=outputs['logits_delta'],
                binary_logits=outputs['binary_logits'],
                ck_real_logits=outputs['ck_real_logits'],
                targets=bx_lbl,
                logits_base=bx_logits,
                class_priors=class_priors_t,
                class_weights=class_weights_t,
                tau_logit_adjust=config_data['tau_logit_adjust'],
                binary_loss_weight=config_data['binary_loss_weight'],
                ck_real_loss_weight=config_data['ck_real_loss_weight'],
                kl_anchor_weight=config_data['kl_anchor_weight'],
                residual_reg_weight=config_data['residual_reg_weight']
            )
            
            loss_total = loss_outputs['loss_total']
            loss_total.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size_curr = len(bx_lbl)
            epoch_train_losses["loss_total"] += loss_outputs["loss_total"].item() * batch_size_curr
            epoch_train_losses["loss_6way"] += loss_outputs["loss_6way"].item() * batch_size_curr
            epoch_train_losses["loss_binary"] += loss_outputs["loss_binary"].item() * batch_size_curr
            epoch_train_losses["loss_ck_real"] += loss_outputs["loss_ck_real"].item() * batch_size_curr
            epoch_train_losses["loss_kl"] += loss_outputs["loss_kl"].item() * batch_size_curr
            epoch_train_losses["loss_residual"] += loss_outputs["loss_residual"].item() * batch_size_curr
            total_samples += batch_size_curr
            
        for k in epoch_train_losses:
            epoch_train_losses[k] /= total_samples
            
        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []
        val_c_probs = []
        val_y_ck_list = []
        
        with torch.no_grad():
            for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_y_ck, bx_logits in val_loader:
                bx_text = bx_text.to(device)
                bx_img_g = bx_img_g.to(device)
                bx_img_p = bx_img_p.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)
                bx_logits = bx_logits.to(device)
                
                outputs = model(
                    text_features=bx_text,
                    image_global_features=bx_img_g,
                    image_patch_features=bx_img_p,
                    kg_features=bx_kg,
                    relation_ids=bx_rel,
                    baseline_logits=bx_logits,
                    ablation_no_c_emb=True
                )
                logits_final = outputs['logits_final']
                c_logit = outputs['c_logit']
                
                preds = torch.argmax(logits_final, dim=-1).cpu().numpy()
                probs_tvcs = torch.sigmoid(c_logit).cpu().numpy()
                
                val_preds.extend(preds)
                val_targets.extend(bx_lbl.numpy())
                val_c_probs.extend(probs_tvcs)
                val_y_ck_list.extend(bx_y_ck.numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_c_probs = np.array(val_c_probs)
        val_y_ck_arr = np.array(val_y_ck_list)
        
        # Calculate Validation Metrics
        acc = accuracy_score(val_targets, val_preds)
        micro_f1 = f1_score(val_targets, val_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(6)), zero_division=0)
        ck_f1 = per_class_f1[2]
        
        tvcs_mask = (val_y_ck_arr != -1)
        if tvcs_mask.sum() > 0 and len(np.unique(val_y_ck_arr[tvcs_mask])) > 1:
            tvcs_auc = roc_auc_score(val_y_ck_arr[tvcs_mask], val_c_probs[tvcs_mask])
            tvcs_pr_auc = average_precision_score(val_y_ck_arr[tvcs_mask], val_c_probs[tvcs_mask])
        else:
            tvcs_auc = 0.5
            tvcs_pr_auc = 0.5
            
        mask_c_real = (val_y_ck_arr == 0)
        mask_c_ck = (val_y_ck_arr == 1)
        mean_c_real = float(np.mean(val_c_probs[mask_c_real])) if mask_c_real.sum() > 0 else 0.0
        mean_c_ck = float(np.mean(val_c_probs[mask_c_ck])) if mask_c_ck.sum() > 0 else 0.0
        tvcs_delta = mean_c_ck - mean_c_real
        
        selection_score = 0.45 * macro_f1 + 0.35 * ck_f1 + 0.20 * tvcs_auc
        
        is_best = (selection_score > best_score)
        if is_best:
            best_score = selection_score
            best_epoch_num = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': selection_score,
                'val_metrics': {
                    'accuracy': acc,
                    'micro_f1': micro_f1,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'ck_f1': ck_f1,
                    'tvcs_auc': tvcs_auc,
                    'tvcs_pr_auc': tvcs_pr_auc,
                    'mean_c_real': mean_c_real,
                    'mean_c_ck': mean_c_ck,
                    'tvcs_delta': tvcs_delta
                },
                'config': config_data
            }
            torch.save(checkpoint_state, os.path.join(ckpt_dir, ckpt_name))
            print(f"  [+] Saved best checkpoint with Val Score {selection_score:.4f}")
        else:
            patience_counter += 1
            
        epoch_log_entry = {
            "epoch": epoch + 1,
            "config_name": config_name,
            "train_total_loss": epoch_train_losses["loss_total"],
            "train_6way_loss": epoch_train_losses["loss_6way"],
            "train_binary_loss": epoch_train_losses["loss_binary"],
            "train_ck_real_loss": epoch_train_losses["loss_ck_real"],
            "train_kl_anchor_loss": epoch_train_losses["loss_kl"],
            "train_residual_reg_loss": epoch_train_losses["loss_residual"],
            "val_accuracy": acc,
            "val_micro_f1": micro_f1,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "val_ck_f1": ck_f1,
            "val_tvcs_auc": tvcs_auc,
            "val_tvcs_pr_auc": tvcs_pr_auc,
            "val_mean_c_real": mean_c_real,
            "val_mean_c_ck": mean_c_ck,
            "val_tvcs_delta": tvcs_delta,
            "val_selection_score": selection_score,
            "best_epoch": is_best
        }
        epoch_logs.append(epoch_log_entry)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_train_losses['loss_total']:.4f} | "
              f"Val Score: {selection_score:.4f} (Acc: {acc:.4f}, Macro: {macro_f1:.4f}, CK-F1: {ck_f1:.4f}, AUC: {tvcs_auc:.4f}) | "
              f"Best: {is_best}")
              
        if patience_counter >= patience:
            print(f"[-] Early stopping triggered. Best epoch was {best_epoch_num} with Val Score {best_score:.4f}")
            break
            
    # 11. Save epoch log specifically for this config
    clean_config_name = config_name.replace('-', '_').lower()
    epoch_csv_path = os.path.join(out_dir, f"I_BC_EPOCH_LOG_{clean_config_name}.csv")
    df_epoch = pd.DataFrame(epoch_logs)
    df_epoch.to_csv(epoch_csv_path, index=False)
    print(f"[+] Saved epoch training log to: {epoch_csv_path}")
    
    # 12. Append / Update best epoch row in I_BC_VAL_METRICS_ALL_CONFIGS.csv
    best_entry = next(entry for entry in epoch_logs if entry["epoch"] == best_epoch_num)
    # Copy and clean the best_epoch flag to True for the final selection metrics
    best_row_data = best_entry.copy()
    best_row_data["best_epoch"] = True
    
    report_csv = os.path.join(out_dir, "I_BC_VAL_METRICS_ALL_CONFIGS.csv")
    if os.path.exists(report_csv):
        df_all = pd.read_csv(report_csv)
        # Drop existing row for this config to avoid duplication
        df_all = df_all[df_all["config_name"] != config_name]
        df_all = pd.concat([df_all, pd.DataFrame([best_row_data])], ignore_index=True)
    else:
        df_all = pd.DataFrame([best_row_data])
        
    df_all.to_csv(report_csv, index=False)
    print(f"[+] Updated aggregated config validation metrics in: {report_csv}")
    
    # 13. Regenerate Training Summary and Final Decision files dynamically
    regenerate_global_reports(args.project_root)
    print("[+] Regenerated global training summary and decision reports.")

if __name__ == "__main__":
    main()
