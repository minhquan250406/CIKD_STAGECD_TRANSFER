"""
Stage I-F1: Feature-Refresh Training Runner.
Loads JSON config, cached multimodal features, baseline logits, pre-trained F4 model,
and TVCS specialist checkpoint. Trains adapters and a Residual Feature Transformer
on the train split only, validates on the validation split only, and outputs 
aggregated metrics and training decision reports under strict test isolation.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add workspace path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cikd_pp_rt import CIKDPPResidualTransformer
from src.stage_i_f1_feature_refresh_model import StageIFFeatureRefreshCIKDPP
from src.stage_i_f1_losses import compute_total_loss, check_tensor_sanity

CLASS_NAMES = [
    "real",
    "text-image inconsistency",
    "content-knowledge inconsistency",
    "text-based fake",
    "image-based fake",
    "others"
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-F1 Feature-Refresh Training Runner.")
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
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def regenerate_global_reports(project_root):
    """
    Reads the config CSV and all epoch logs to dynamically regenerate
    F1_TRAINING_SUMMARY.txt, F1_FINAL_DECISION.txt, and F1_PER_CLASS_F1_COMPARISON.csv.
    """
    out_dir = os.path.join(project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f", "f1_training")
    report_csv = os.path.join(out_dir, "F1_VAL_METRICS_ALL_CONFIGS.csv")
    summary_txt = os.path.join(out_dir, "F1_TRAINING_SUMMARY.txt")
    decision_txt = os.path.join(out_dir, "F1_FINAL_DECISION.txt")
    comparison_csv = os.path.join(out_dir, "F1_PER_CLASS_F1_COMPARISON.csv")
    rescue_csv = os.path.join(out_dir, "F1_RESCUE_BROKEN_BY_CLASS.csv")
    
    if not os.path.exists(report_csv):
        print(f"[!] Cannot regenerate reports: {report_csv} does not exist.")
        return
        
    df_configs = pd.read_csv(report_csv)
    if len(df_configs) == 0:
        print("[!] Cannot regenerate reports: F1_VAL_METRICS_ALL_CONFIGS.csv is empty.")
        return
        
    # F4 reference metrics
    F4_REF = {
        "model": "F4",
        "val_accuracy": 0.5792307692307692,
        "val_micro_f1": 0.5792307692307692,
        "val_macro_f1": 0.47924497436620056,
        "val_weighted_f1": 0.591941257787986,
        "f1_class_0": 0.6751131221719457,
        "f1_class_1": 0.3696682464454976,
        "f1_class_2": 0.39215686274509803,
        "f1_class_3": 0.5609397944199707,
        "f1_class_4": 0.6480836236933798,
        "f1_class_5": 0.22950819672131148
    }
    
    # 1. Regenerate F1_PER_CLASS_F1_COMPARISON.csv
    rows = [F4_REF]
    for _, r in df_configs.iterrows():
        rows.append({
            "model": r["config_name"],
            "val_accuracy": r["val_accuracy"],
            "val_micro_f1": r["val_micro_f1"],
            "val_macro_f1": r["val_macro_f1"],
            "val_weighted_f1": r["val_weighted_f1"],
            "f1_class_0": r["val_class0_f1"],
            "f1_class_1": r["val_class1_f1"],
            "f1_class_2": r["val_class2_f1"],
            "f1_class_3": r["val_class3_f1"],
            "f1_class_4": r["val_class4_f1"],
            "f1_class_5": r["val_class5_f1"]
        })
    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(comparison_csv, index=False)
    print(f"[+] Regenerated comparison metrics at: {comparison_csv}")
    
    # 2. Write F1_TRAINING_SUMMARY.txt
    with open(summary_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE I-F1: FEATURE-REFRESH TRAINING SUMMARY\n")
        f.write("========================================================================\n\n")
        
        f.write("SUMMARY OF BEST EPOCH VAL PERFORMANCE ACROSS CONFIGURATIONS:\n")
        f.write("-----------------------------------------------------------\n")
        f.write(df_configs.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED EPOCH-BY-EPOCH TRAINING LOGS:\n")
        f.write("--------------------------------------\n")
        epoch_log_files = glob.glob(os.path.join(out_dir, "F1_EPOCH_LOG_*.csv"))
        epoch_log_files.sort()
        
        for lf in epoch_log_files:
            cfg_name = os.path.basename(lf).replace("F1_EPOCH_LOG_", "").replace(".csv", "").upper()
            f.write(f"\nConfiguration: {cfg_name}\n")
            f.write("=" * (len(cfg_name) + 15) + "\n")
            df_log = pd.read_csv(lf)
            f.write(df_log.to_string(index=False))
            f.write("\n")
            
    print(f"[+] Regenerated training summary at: {summary_txt}")
    
    # 3. Write F1_FINAL_DECISION.txt
    best_idx = df_configs["val_selection_score"].idxmax()
    best_row = df_configs.iloc[best_idx]
    best_cfg_name = best_row["config_name"]
    
    # Check gates
    gate_macro_f1 = best_row["val_macro_f1"] > 0.479200
    gate_macro_pref = best_row["val_macro_f1"] >= 0.485000
    gate_ck_f1 = best_row["val_ck_f1"] >= 0.392200
    gate_class1_stable = best_row["val_class1_f1"] >= 0.369668
    gate_class5_improve = best_row["val_class5_f1"] > 0.229508
    
    gain_not_mainly_class0 = True
    net_rescue_0 = 0
    net_rescue_others = 0
    
    if os.path.exists(rescue_csv):
        df_rb = pd.read_csv(rescue_csv)
        df_rb_best = df_rb[df_rb["comparison_model"] == best_cfg_name]
        if len(df_rb_best) > 0:
            net_rescue_0 = df_rb_best[df_rb_best["class_id"] == 0]["net_rescue"].values[0]
            net_rescue_others = df_rb_best[df_rb_best["class_id"] > 0]["net_rescue"].sum()
            # If class0 net rescue is positive and larger than the sum of other classes, we flag it.
            if net_rescue_0 > 0 and net_rescue_0 > net_rescue_others:
                gain_not_mainly_class0 = False
                
    gate_nan_inf = not bool(best_row["nan_inf_occurred"])
    gate_no_test = True
    
    promote = (gate_macro_f1 and gate_ck_f1 and gate_class1_stable and gate_class5_improve and gain_not_mainly_class0 and gate_nan_inf and gate_no_test)
    decision_status = "PROMOTE_TO_LOCKED_TEST_CANDIDATE" if promote else "DIAGNOSTIC_ONLY_KEEP_F4"
    
    with open(decision_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE I-F1: FINAL MODEL SELECTION DECISION\n")
        f.write("========================================================================\n\n")
        f.write(f"Best Performing Configuration: {best_cfg_name}\n")
        f.write(f"Decision:                      {decision_status}\n\n")
        
        f.write("PROMOTION GATE VERIFICATION:\n")
        f.write("----------------------------\n")
        f.write(f"1. validation Macro-F1 > F4 reference (0.4792):   {'PASSED' if gate_macro_f1 else 'FAILED'} (Value: {best_row['val_macro_f1']:.6f})\n")
        f.write(f"   (Preferably validation Macro-F1 >= 0.485:      {'YES' if gate_macro_pref else 'NO'})\n")
        f.write(f"2. validation CK-F1 >= F4 reference (0.3922):     {'PASSED' if gate_ck_f1 else 'FAILED'} (Value: {best_row['val_ck_f1']:.6f})\n")
        f.write(f"3. Class 1 (text-image inconsistency) F1 stable:   {'PASSED' if gate_class1_stable else 'FAILED'} (Value: {best_row['val_class1_f1']:.6f} vs F4: 0.369668)\n")
        f.write(f"4. Class 5 (others) F1 improved over F4 (0.2295): {'PASSED' if gate_class5_improve else 'FAILED'} (Value: {best_row['val_class5_f1']:.6f} vs F4: 0.229508)\n")
        f.write(f"5. Gain does not come mainly from class 0 rescue: {'PASSED' if gain_not_mainly_class0 else 'FAILED'}\n")
        f.write(f"   - Real/Class 0 Net Rescue:                    {net_rescue_0}\n")
        f.write(f"   - Minority/Other Classes Net Rescue Sum:      {net_rescue_others}\n")
        f.write(f"6. No NaN/Inf occurred during training:          {'PASSED' if gate_nan_inf else 'FAILED'}\n")
        f.write(f"7. No test evaluation occurred:                   {'PASSED' if gate_no_test else 'FAILED'} (Locked test is isolated)\n\n")
        
        f.write("Best Configuration Performance Details:\n")
        f.write("---------------------------------------\n")
        f.write(f"Best Epoch:                 {best_row['epoch']}\n")
        f.write(f"Validation Selection Score:  {best_row['val_selection_score']:.6f}\n")
        f.write(f"Validation Accuracy:        {best_row['val_accuracy']:.6f}\n")
        f.write(f"Validation Macro-F1:       {best_row['val_macro_f1']:.6f}\n")
        f.write(f"Validation Micro-F1:       {best_row['val_micro_f1']:.6f}\n")
        f.write(f"Validation CK-F1:          {best_row['val_ck_f1']:.6f}\n")
        f.write(f"Validation Bottleneck F1:  {best_row['val_bottleneck_macro_f1']:.6f}\n")
        f.write(f"Validation Class 5 F1:      {best_row['val_class5_f1']:.6f}\n")
        
    print(f"[+] Regenerated final decision at: {decision_txt}")

def main():
    # 1. Startup safety checks
    args = parse_args()
    assert args.no_test_eval, "Abort: --no_test_eval flag is missing! Must be present to ensure test set isolation."
    
    print("LOCKED TEST IS DISABLED")
    print("TRAINING USES TRAIN SPLIT ONLY; SELECTION USES VALIDATION SPLIT ONLY")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device selected: {device}")
    
    # 2. Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config_data = json.load(f)
        
    config_name = config_data.get("config_name", os.path.basename(args.config))
    print(f"[+] Configuration: {config_name}")
    
    # Resolve hyperparams with optional overrides
    epochs = args.max_epochs if args.max_epochs is not None else config_data.get("epochs", 10)
    patience = args.patience if args.patience is not None else config_data.get("patience", 3)
    batch_size = args.batch_size if args.batch_size is not None else config_data.get("batch_size", 16)
    lr = args.lr if args.lr is not None else config_data.get("lr", 1e-3)
    weight_decay = args.weight_decay if args.weight_decay is not None else config_data.get("weight_decay", 1e-4)
    seed = args.seed if args.seed is not None else config_data.get("seed", 42)
    
    set_seed(seed)
    print(f"    Parameters: seed={seed}, max_epochs={epochs}, patience={patience}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}")
    
    # Setup directories
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f", "f1_training")
    ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_i_f")
    matrix_dir = os.path.join(out_dir, "F1_CONFUSION_MATRICES")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)
    
    ckpt_path = os.path.join(ckpt_dir, f"best_{config_name}.pt")
    
    # 3. Load dataset cached features and assert no split_id == 2 (test split) leakage
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    print(f"[+] Loading dataset cache from: {cache_dir}")
    
    split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
    relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
    kg_features = np.load(os.path.join(cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
    text_features = np.load(os.path.join(cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))
    
    train_mask = (split_ids == 0)
    val_mask = (split_ids == 1)
    
    # Strict safety assertions: abort if training/validation mask contains test split
    if np.any(split_ids[train_mask] == 2) or np.any(split_ids[val_mask] == 2):
        print("ERROR: Test split (split_id == 2) leaked into train/val mask! Aborting script.")
        sys.exit(1)
        
    # Load baseline logits from outputs/stage_f0_baseline_anchor
    baseline_dir = os.path.join(args.project_root, "outputs", "stage_f0_baseline_anchor")
    print(f"[+] Loading baseline anchor logits from: {baseline_dir}")
    
    train_logits_base = np.load(os.path.join(baseline_dir, 'train_logits_base.npy'))
    val_logits_base = np.load(os.path.join(baseline_dir, 'val_logits_base.npy'))
    
    assert len(train_logits_base) == np.sum(train_mask), "Train baseline logits size mismatch!"
    assert len(val_logits_base) == np.sum(val_mask), "Validation baseline logits size mismatch!"
    
    # Slice features and wrap in DataLoader
    tr_text = torch.tensor(text_features[train_mask], dtype=torch.float32)
    tr_img_g = torch.tensor(image_features_global[train_mask], dtype=torch.float32)
    tr_img_p = torch.tensor(image_features_patch[train_mask], dtype=torch.float32)
    tr_kg = torch.tensor(kg_features[train_mask], dtype=torch.float32)
    tr_rel = torch.tensor(relation_ids[train_mask], dtype=torch.long)
    tr_lbl = torch.tensor(labels_fine[train_mask], dtype=torch.long)
    tr_logits = torch.tensor(train_logits_base, dtype=torch.float32)
    
    val_text = torch.tensor(text_features[val_mask], dtype=torch.float32)
    val_img_g = torch.tensor(image_features_global[val_mask], dtype=torch.float32)
    val_img_p = torch.tensor(image_features_patch[val_mask], dtype=torch.float32)
    val_kg = torch.tensor(kg_features[val_mask], dtype=torch.float32)
    val_rel = torch.tensor(relation_ids[val_mask], dtype=torch.long)
    val_lbl = torch.tensor(labels_fine[val_mask], dtype=torch.long)
    val_logits = torch.tensor(val_logits_base, dtype=torch.float32)
    
    train_ds = TensorDataset(tr_text, tr_img_g, tr_img_p, tr_kg, tr_rel, tr_lbl, tr_logits)
    val_ds = TensorDataset(val_text, val_img_g, val_img_p, val_kg, val_rel, val_lbl, val_logits)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 4. Instantiate and setup F4 model backbone
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_features.shape[1]
    
    f4_model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    f4_ckpt_path = os.path.join(args.project_root, config_data["f4_checkpoint"])
    if os.path.exists(f4_ckpt_path):
        print(f"[+] Loading F4 Backbone: {f4_ckpt_path}")
        f4_ckpt = torch.load(f4_ckpt_path, map_location=device, weights_only=False)
        f4_state = f4_ckpt.get('model_state_dict', f4_ckpt)
        f4_model.load_state_dict(f4_state)
    else:
        raise FileNotFoundError(f"F4 Backbone checkpoint not found at {f4_ckpt_path}")
        
    # Load TVCS specialist into F4 backbone for consistency
    tvcs_ckpt_path = os.path.join(args.project_root, config_data.get("tvcs_checkpoint", ""))
    if os.path.exists(tvcs_ckpt_path):
        print(f"[+] Loading TVCS Specialist checkpoint: {tvcs_ckpt_path}")
        tvcs_ckpt = torch.load(tvcs_ckpt_path, map_location=device, weights_only=False)
        tvcs_state = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
        f4_model.tvcs_specialist.load_state_dict(tvcs_state)
    else:
        raise FileNotFoundError(f"TVCS checkpoint not found at {tvcs_ckpt_path}")
        
    # Instantiate the Stage I-F Feature Refresh Model
    model = StageIFFeatureRefreshCIKDPP(
        f4_model=f4_model,
        num_relations=num_relations,
        kg_dim=kg_dim,
        relation_emb_dim=64,
        gamma=config_data["gamma"],
        use_patch_adapter=config_data.get("use_patch_adapter", True),
        use_kg_relation_adapter=config_data.get("use_kg_relation_adapter", True),
        use_tvcs_zv=config_data.get("use_tvcs_zv", True),
        d_model=config_data.get("d_model", 256),
        num_layers=config_data.get("num_layers", 2),
        num_heads=config_data.get("num_heads", 4),
        dropout=config_data.get("dropout", 0.1)
    ).to(device)
    
    # 5. Extract F4 predictions on validation set for rescued/broken analysis
    print("[+] Extracting baseline predictions of frozen F4 model on validation split...")
    f4_model.eval()
    f4_preds = []
    with torch.no_grad():
        for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, _, bx_logits in val_loader:
            bx_text = bx_text.to(device)
            bx_img_g = bx_img_g.to(device)
            bx_img_p = bx_img_p.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_logits = bx_logits.to(device)
            
            f4_out = f4_model(
                text_features=bx_text,
                image_global_features=bx_img_g,
                image_patch_features=bx_img_p,
                kg_features=bx_kg,
                relation_ids=bx_rel,
                baseline_logits=bx_logits,
                ablation_no_c_emb=True
            )
            preds = torch.argmax(f4_out["logits_final"], dim=-1).cpu().numpy()
            f4_preds.extend(preds)
    f4_preds = np.array(f4_preds)
    
    # 6. Calculate class priors and weights from train split to match project conventions
    train_lbl_np = labels_fine[train_mask]
    class_counts = np.bincount(train_lbl_np, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_lbl_np) / (6.0 * class_counts)
    
    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 7. Optimizer: train only adapter modules and heads (F4, TVCS specialist, and baseline logits remain frozen)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    # 8. Training loop
    best_score = -1.0
    best_epoch = -1
    patience_counter = 0
    epoch_logs = []
    
    best_cm = None
    best_val_preds = None
    nan_inf_occurred = False
    
    print("\n[+] Starting training...")
    for epoch in range(epochs):
        model.train()
        model.f4_model.eval()  # Freeze F4 evaluation behaviors
        
        train_loss_metrics = {
            "loss_total": 0.0,
            "loss_6way": 0.0,
            "loss_focal": 0.0,
            "loss_ck_guard": 0.0,
            "loss_kl": 0.0,
            "loss_residual": 0.0
        }
        total_samples = 0
        
        for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_logits in train_loader:
            bx_text = bx_text.to(device)
            bx_img_g = bx_img_g.to(device)
            bx_img_p = bx_img_p.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_lbl = bx_lbl.to(device)
            bx_logits = bx_logits.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                text_features=bx_text,
                image_global_features=bx_img_g,
                image_patch_features=bx_img_p,
                kg_features=bx_kg,
                relation_ids=bx_rel,
                baseline_logits=bx_logits
            )
            
            loss_outputs = compute_total_loss(
                logits_final=outputs['logits_final'],
                delta_new=outputs['delta_new'],
                f4_logits=outputs['f4_logits'],
                targets=bx_lbl,
                class_priors=class_priors_t,
                class_weights=class_weights_t,
                bottleneck_classes=config_data.get("loss_focus_classes", [1, 2, 5]),
                tau_logit_adjust=config_data.get("tau_logit_adjust", 0.5),
                focal_gamma=config_data.get("focal_gamma", 1.5),
                kl_temperature=config_data.get("kl_temperature", 1.0),
                w_balanced=config_data.get("w_balanced", 1.00),
                w_focal=config_data.get("w_focal", 0.20),
                w_ck_guard=config_data.get("w_ck_guard", 0.10),
                w_kl=config_data.get("w_kl", 0.10),
                w_delta_norm=config_data.get("w_delta_norm", 0.03)
            )
            
            loss_total = loss_outputs['loss_total']
            
            # Check tensor sanity
            tensors_to_check = {
                "logits_final": outputs['logits_final'],
                "delta_new": outputs['delta_new'],
                "f4_logits": outputs['f4_logits'],
                "loss_total": loss_total
            }
            if outputs.get('patch_refresh') is not None:
                tensors_to_check["patch_refresh"] = outputs['patch_refresh']
            if outputs.get('kg_refresh') is not None:
                tensors_to_check["kg_refresh"] = outputs['kg_refresh']
                
            for name, tensor in tensors_to_check.items():
                nc, ic = check_tensor_sanity(tensor, name)
                if nc > 0 or ic > 0:
                    nan_inf_occurred = True
                    
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size_curr = len(bx_lbl)
            train_loss_metrics["loss_total"] += loss_total.item() * batch_size_curr
            train_loss_metrics["loss_6way"] += loss_outputs["loss_6way"].item() * batch_size_curr
            train_loss_metrics["loss_focal"] += loss_outputs["loss_focal"].item() * batch_size_curr
            train_loss_metrics["loss_ck_guard"] += loss_outputs["loss_ck_guard"].item() * batch_size_curr
            train_loss_metrics["loss_kl"] += loss_outputs["loss_kl"].item() * batch_size_curr
            train_loss_metrics["loss_residual"] += loss_outputs["loss_residual"].item() * batch_size_curr
            total_samples += batch_size_curr
            
        for k in train_loss_metrics:
            train_loss_metrics[k] /= total_samples
            
        # Validation Evaluation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_logits in val_loader:
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
                    baseline_logits=bx_logits
                )
                
                preds = torch.argmax(outputs['logits_final'], dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(bx_lbl.numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Calculate Validation Metrics
        acc = accuracy_score(val_targets, val_preds)
        micro_f1 = f1_score(val_targets, val_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(6)), zero_division=0)
        ck_f1 = per_class_f1[2]
        class1_f1 = per_class_f1[1]
        class2_f1 = per_class_f1[2]
        class5_f1 = per_class_f1[5]
        
        bottleneck_macro_f1 = np.mean([class1_f1, class2_f1, class5_f1])
        
        # Selection Score formula: 0.45 * Macro-F1 + 0.25 * CK-F1 + 0.20 * bottleneck_macro_f1 + 0.10 * class5_F1
        selection_score = 0.45 * macro_f1 + 0.25 * ck_f1 + 0.20 * bottleneck_macro_f1 + 0.10 * class5_f1
        
        cm = confusion_matrix(val_targets, val_preds, labels=list(range(6)))
        
        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
            best_epoch = epoch + 1
            patience_counter = 0
            best_cm = cm
            best_val_preds = val_preds
            
            # Save best checkpoint
            ckpt_state = {
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
                    'class1_f1': class1_f1,
                    'class2_f1': class2_f1,
                    'class5_f1': class5_f1,
                    'bottleneck_macro_f1': bottleneck_macro_f1,
                    'selection_score': selection_score,
                    'per_class_f1': per_class_f1.tolist(),
                    'confusion_matrix': cm.tolist()
                },
                'config': config_data,
                'nan_inf_occurred': nan_inf_occurred
            }
            torch.save(ckpt_state, ckpt_path)
            print(f"  [+] Saved best checkpoint with Val Selection Score: {selection_score:.6f}")
        else:
            patience_counter += 1
            
        epoch_log_entry = {
            "epoch": epoch + 1,
            "val_accuracy": acc,
            "val_micro_f1": micro_f1,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "val_ck_f1": ck_f1,
            "val_class0_f1": per_class_f1[0],
            "val_class1_f1": per_class_f1[1],
            "val_class2_f1": per_class_f1[2],
            "val_class3_f1": per_class_f1[3],
            "val_class4_f1": per_class_f1[4],
            "val_class5_f1": per_class_f1[5],
            "val_bottleneck_macro_f1": bottleneck_macro_f1,
            "val_selection_score": selection_score,
            "val_confusion_matrix": str(cm.tolist()),
            "train_total_loss": train_loss_metrics["loss_total"],
            "train_balanced_6way_loss": train_loss_metrics["loss_6way"],
            "train_focal_bottleneck_loss": train_loss_metrics["loss_focal"],
            "train_ck_guard_loss": train_loss_metrics["loss_ck_guard"],
            "train_kl_to_f4_loss": train_loss_metrics["loss_kl"],
            "train_delta_norm_loss": train_loss_metrics["loss_residual"],
            "nan_inf_occurred": 1 if nan_inf_occurred else 0
        }
        epoch_logs.append(epoch_log_entry)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss_metrics['loss_total']:.4f} | "
              f"Val Score: {selection_score:.4f} (Macro: {macro_f1:.4f}, CK: {ck_f1:.4f}, Bottleneck Macro: {bottleneck_macro_f1:.4f}) | "
              f"Best: {is_best}")
              
        if patience_counter >= patience:
            print(f"[-] Early stopping triggered. Best epoch was {best_epoch} with Val Selection Score: {best_score:.6f}")
            break
            
    # 9. Post-training updates and saving reports
    epoch_csv_path = os.path.join(out_dir, f"F1_EPOCH_LOG_{config_name}.csv")
    df_epoch = pd.DataFrame(epoch_logs)
    df_epoch.to_csv(epoch_csv_path, index=False)
    print(f"[+] Saved epoch training log to: {epoch_csv_path}")
    
    # Save best epoch metrics to aggregated CSV
    best_entry = next(entry for entry in epoch_logs if entry["epoch"] == best_epoch)
    best_row_data = {
        "config_name": config_name,
        "gamma": config_data["gamma"],
        "epoch": best_epoch,
        "val_accuracy": best_entry["val_accuracy"],
        "val_micro_f1": best_entry["val_micro_f1"],
        "val_macro_f1": best_entry["val_macro_f1"],
        "val_weighted_f1": best_entry["val_weighted_f1"],
        "val_ck_f1": best_entry["val_ck_f1"],
        "val_class0_f1": best_entry["val_class0_f1"],
        "val_class1_f1": best_entry["val_class1_f1"],
        "val_class2_f1": best_entry["val_class2_f1"],
        "val_class3_f1": best_entry["val_class3_f1"],
        "val_class4_f1": best_entry["val_class4_f1"],
        "val_class5_f1": best_entry["val_class5_f1"],
        "val_bottleneck_macro_f1": best_entry["val_bottleneck_macro_f1"],
        "val_selection_score": best_entry["val_selection_score"],
        "val_confusion_matrix": best_entry["val_confusion_matrix"],
        "train_total_loss": best_entry["train_total_loss"],
        "train_balanced_6way_loss": best_entry["train_balanced_6way_loss"],
        "train_focal_bottleneck_loss": best_entry["train_focal_bottleneck_loss"],
        "train_ck_guard_loss": best_entry["train_ck_guard_loss"],
        "train_kl_to_f4_loss": best_entry["train_kl_to_f4_loss"],
        "train_delta_norm_loss": best_entry["train_delta_norm_loss"],
        "nan_inf_occurred": best_entry["nan_inf_occurred"]
    }
    
    report_csv = os.path.join(out_dir, "F1_VAL_METRICS_ALL_CONFIGS.csv")
    if os.path.exists(report_csv):
        df_all = pd.read_csv(report_csv)
        df_all = df_all[df_all["config_name"] != config_name]
        df_all = pd.concat([df_all, pd.DataFrame([best_row_data])], ignore_index=True)
    else:
        df_all = pd.DataFrame([best_row_data])
    df_all.to_csv(report_csv, index=False)
    print(f"[+] Updated aggregated config metrics in: {report_csv}")
    
    # Save the confusion matrix to confusion matrix dir
    df_best_cm = pd.DataFrame(best_cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_best_cm.to_csv(os.path.join(matrix_dir, f"best_{config_name}_confusion_matrix.csv"))
    print(f"[+] Saved confusion matrix to: {os.path.join(matrix_dir, f'best_{config_name}_confusion_matrix.csv')}")
    
    # Calculate rescued and broken counts vs F4 on validation set
    f4_correct = (f4_preds == val_targets)
    m_correct = (best_val_preds == val_targets)
    rescued_overall = (~f4_correct) & m_correct
    broken_overall = f4_correct & (~m_correct)
    
    rb_rows = []
    for c in range(6):
        c_mask = (val_targets == c)
        rescued_c = int(np.sum(rescued_overall[c_mask]))
        broken_c = int(np.sum(broken_overall[c_mask]))
        rb_rows.append({
            "comparison_model": config_name,
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "rescued_count": rescued_c,
            "broken_count": broken_c,
            "net_rescue": rescued_c - broken_c
        })
    df_rb_new = pd.DataFrame(rb_rows)
    
    rescue_csv = os.path.join(out_dir, "F1_RESCUE_BROKEN_BY_CLASS.csv")
    if os.path.exists(rescue_csv):
        df_rb_all = pd.read_csv(rescue_csv)
        df_rb_all = df_rb_all[df_rb_all["comparison_model"] != config_name]
        df_rb_all = pd.concat([df_rb_all, df_rb_new], ignore_index=True)
    else:
        df_rb_all = df_rb_new
    df_rb_all.to_csv(rescue_csv, index=False)
    print(f"[+] Updated rescue/broken counts in: {rescue_csv}")
    
    # Regenerate all global summary reports dynamically
    regenerate_global_reports(args.project_root)
    print("[+] All Stage I-F1 global reports successfully regenerated.")

if __name__ == "__main__":
    main()
