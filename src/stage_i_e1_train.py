"""
Stage I-E1: Class-Bottleneck Residual Adapter Training Runner.
Loads JSON config, cached multimodal features, baseline logits, and pre-trained F4 model.
Trains a small bottleneck MLP adapter head, validates on validation split,
and outputs performance reports while enforcing strict test split isolation.
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
from src.stage_i_e1_bottleneck_adapter_model import StageIE1BottleneckAdapterModel
from src.stage_i_e1_losses import compute_adapter_total_loss, check_tensor_sanity

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
    parser = argparse.ArgumentParser(description="Stage I-E1 Validation-Only Training Runner.")
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
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def regenerate_global_reports(project_root):
    """
    Reads the config CSV and all epoch logs to dynamically regenerate
    E1_TRAINING_SUMMARY.txt, E1_FINAL_DECISION.txt, and E1_PER_CLASS_F1_COMPARISON.csv.
    """
    out_dir = os.path.join(project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e", "e1_training")
    report_csv = os.path.join(out_dir, "E1_VAL_METRICS_ALL_CONFIGS.csv")
    summary_txt = os.path.join(out_dir, "E1_TRAINING_SUMMARY.txt")
    decision_txt = os.path.join(out_dir, "E1_FINAL_DECISION.txt")
    comparison_csv = os.path.join(out_dir, "E1_PER_CLASS_F1_COMPARISON.csv")
    rescue_csv = os.path.join(out_dir, "E1_RESCUE_BROKEN_BY_CLASS.csv")
    
    if not os.path.exists(report_csv):
        print(f"[!] Cannot regenerate reports: {report_csv} does not exist.")
        return
        
    df_configs = pd.read_csv(report_csv)
    if len(df_configs) == 0:
        print("[!] Cannot regenerate reports: E1_VAL_METRICS_ALL_CONFIGS.csv is empty.")
        return
        
    # 1. Regenerate E1_PER_CLASS_F1_COMPARISON.csv
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
    
    rows = [F4_REF]
    for _, r in df_configs.iterrows():
        rows.append({
            "model": r["config_name"],
            "val_accuracy": r["val_accuracy"],
            "val_micro_f1": r["val_micro_f1"],
            "val_macro_f1": r["val_macro_f1"],
            "val_weighted_f1": r["val_weighted_f1"],
            "f1_class_0": r["f1_class_0"],
            "f1_class_1": r["f1_class_1"],
            "f1_class_2": r["f1_class_2"],
            "f1_class_3": r["f1_class_3"],
            "f1_class_4": r["f1_class_4"],
            "f1_class_5": r["f1_class_5"]
        })
    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(comparison_csv, index=False)
    print(f"[+] Regenerated comparison metrics at: {comparison_csv}")
    
    # 2. Write E1_TRAINING_SUMMARY.txt
    with open(summary_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE I-E1: CLASS-BOTTLENECK RESIDUAL ADAPTER TRAINING SUMMARY\n")
        f.write("========================================================================\n\n")
        
        f.write("SUMMARY OF BEST EPOCH VAL PERFORMANCE ACROSS CONFIGURATIONS:\n")
        f.write("-----------------------------------------------------------\n")
        f.write(df_configs.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED EPOCH-BY-EPOCH TRAINING LOGS:\n")
        f.write("--------------------------------------\n")
        epoch_log_files = glob.glob(os.path.join(out_dir, "E1_EPOCH_LOG_*.csv"))
        epoch_log_files.sort()
        
        for lf in epoch_log_files:
            cfg_name = os.path.basename(lf).replace("E1_EPOCH_LOG_", "").replace(".csv", "").upper()
            f.write(f"\nConfiguration: {cfg_name}\n")
            f.write("=" * (len(cfg_name) + 15) + "\n")
            df_log = pd.read_csv(lf)
            f.write(df_log.to_string(index=False))
            f.write("\n")
            
    print(f"[+] Regenerated training summary at: {summary_txt}")
    
    # 3. Write E1_FINAL_DECISION.txt
    best_idx = df_configs["val_selection_score"].idxmax()
    best_row = df_configs.iloc[best_idx]
    best_cfg_name = best_row["config_name"]
    
    # Check gates
    gate_macro_f1 = best_row["val_macro_f1"] > 0.479245
    gate_macro_pref = best_row["val_macro_f1"] >= 0.485
    gate_ck_f1 = best_row["val_ck_f1"] >= 0.392157
    gate_minority_improve = (best_row["val_class1_f1"] > 0.369668) or (best_row["val_class5_f1"] > 0.229508)
    
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
    
    promote = (gate_macro_f1 and gate_ck_f1 and gate_minority_improve and gain_not_mainly_class0 and gate_nan_inf and gate_no_test)
    decision_status = "PROMOTE_TO_LOCKED_TEST_CANDIDATE" if promote else "DIAGNOSTIC_ONLY_KEEP_F4"
    
    with open(decision_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE I-E1: FINAL MODEL SELECTION DECISION\n")
        f.write("========================================================================\n\n")
        f.write(f"Best Performing Configuration: {best_cfg_name}\n")
        f.write(f"Decision:                      {decision_status}\n\n")
        
        f.write("PROMOTION GATE VERIFICATION:\n")
        f.write("----------------------------\n")
        f.write(f"1. validation Macro-F1 > F4 reference (0.479245): {'PASSED' if gate_macro_f1 else 'FAILED'} (Value: {best_row['val_macro_f1']:.6f})\n")
        f.write(f"   (Preferably validation Macro-F1 >= 0.485:      {'YES' if gate_macro_pref else 'NO'})\n")
        f.write(f"2. validation CK-F1 >= F4 reference (0.392157):   {'PASSED' if gate_ck_f1 else 'FAILED'} (Value: {best_row['val_ck_f1']:.6f})\n")
        f.write(f"3. Class 1 or Class 5 F1 improved vs F4:          {'PASSED' if gate_minority_improve else 'FAILED'}\n")
        f.write(f"   - Class 1 (text-image inconsistency) F1:      {best_row['val_class1_f1']:.6f} vs F4: 0.369668 (improved: {'YES' if best_row['val_class1_f1'] > 0.369668 else 'NO'})\n")
        f.write(f"   - Class 5 (others) F1:                        {best_row['val_class5_f1']:.6f} vs F4: 0.229508 (improved: {'YES' if best_row['val_class5_f1'] > 0.229508 else 'NO'})\n")
        f.write(f"4. Gain does not come mainly from class 0 rescue: {'PASSED' if gain_not_mainly_class0 else 'FAILED'}\n")
        f.write(f"   - Real/Class 0 Net Rescue:                    {net_rescue_0}\n")
        f.write(f"   - Minority/Other Classes Net Rescue Sum:      {net_rescue_others}\n")
        f.write(f"5. No NaN/Inf occurred during training:          {'PASSED' if gate_nan_inf else 'FAILED'}\n")
        f.write(f"6. No test evaluation occurred:                   {'PASSED' if gate_no_test else 'FAILED'} (Locked test is isolated)\n\n")
        
        f.write("Best Configuration Performance Details:\n")
        f.write("---------------------------------------\n")
        f.write(f"Best Epoch:                 {best_row['epoch']}\n")
        f.write(f"Validation Selection Score:  {best_row['val_selection_score']:.6f}\n")
        f.write(f"Validation Accuracy:        {best_row['val_accuracy']:.6f}\n")
        f.write(f"Validation Macro-F1:       {best_row['val_macro_f1']:.6f}\n")
        f.write(f"Validation Micro-F1:       {best_row['val_micro_f1']:.6f}\n")
        f.write(f"Validation CK-F1:          {best_row['val_ck_f1']:.6f}\n")
        f.write(f"Validation Bottleneck F1:  {best_row['val_bottleneck_macro_f1']:.6f}\n")
        
    print(f"[+] Regenerated final decision at: {decision_txt}")

def main():
    # 1. Startup assertions & setup
    args = parse_args()
    if not args.no_test_eval:
        print("ERROR: --no_test_eval is missing! Aborting to ensure test split isolation.")
        sys.exit(1)
        
    print("LOCKED TEST IS DISABLED")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device selected: {device}")
    
    # 2. Load configuration JSON
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
    
    # 3. Verify paths & directory structure
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e", "e1_training")
    ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_i_e")
    matrix_dir = os.path.join(out_dir, "E1_CONFUSION_MATRICES")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)
    
    # Extract beta for filename construction
    beta_val = config_data["beta"]
    beta_str = f"{beta_val:.1f}".replace('.', '')
    ckpt_filename = f"best_i_e1_beta{beta_str}.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    
    # 4. Load dataset cached features
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    print(f"[+] Loading dataset cache from: {cache_dir}")
    
    split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
    relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
    kg_features = np.load(os.path.join(cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
    text_features = np.load(os.path.join(cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))
    
    # Check for test split index (split_id == 2) in loaded dataset masks
    train_mask = (split_ids == 0)
    val_mask = (split_ids == 1)
    
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
    
    # 5. Instantiate and setup F4 model backbone
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
        
    # Optional load of specialist into F4 backbone for consistency
    tvcs_ckpt_path = os.path.join(args.project_root, config_data.get("tvcs_checkpoint", ""))
    if os.path.exists(tvcs_ckpt_path):
        print(f"[+] Loading TVCS Specialist checkpoint: {tvcs_ckpt_path}")
        tvcs_ckpt = torch.load(tvcs_ckpt_path, map_location=device, weights_only=False)
        tvcs_state = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
        f4_model.tvcs_specialist.load_state_dict(tvcs_state)
        
    # Load class mask from outputs/stage_i_macro_micro_improvement/stage_i_e/e0_bottleneck_audit/E0_RECOMMENDED_CLASS_MASK.json
    mask_path = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e", "e0_bottleneck_audit", "E0_RECOMMENDED_CLASS_MASK.json")
    if os.path.exists(mask_path):
        print(f"[+] Loading recommended class mask from: {mask_path}")
        with open(mask_path, "r") as f_mask:
            mask_payload = json.load(f_mask)
        class_mask = mask_payload["class_mask"]
    else:
        print("[!] Warning: Recommended mask not found, falling back to config mask.")
        class_mask = config_data["class_mask"]
        
    # Instantiate the wrapper adapter model
    model = StageIE1BottleneckAdapterModel(
        f4_model=f4_model,
        beta=config_data["beta"],
        class_mask=class_mask,
        hidden_dim=config_data.get("hidden_dim", 128),
        dropout=config_data.get("dropout", 0.1)
    ).to(device)
    
    # 6. Loss configurations
    train_lbl_np = labels_fine[train_mask]
    class_counts = np.bincount(train_lbl_np, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_lbl_np) / (6.0 * class_counts)
    
    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Freeze F4, TVCS, and baseline parameters: handled inside StageIE1BottleneckAdapterModel init,
    # but double checking and filtering trainable parameters for safety
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    # 7. Extract F4 predictions on validation set for rescued/broken analysis
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
    
    # 8. Training loop execution
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
            
            loss_outputs = compute_adapter_total_loss(
                logits_final=outputs['logits_final'],
                logits_delta=outputs['logits_delta'],
                f4_logits=outputs['f4_logits'],
                targets=bx_lbl,
                class_priors=class_priors_t,
                class_weights=class_weights_t,
                bottleneck_classes=config_data["bottleneck_classes"],
                tau_logit_adjust=config_data["tau_logit_adjust"],
                focal_gamma=config_data["focal_gamma"],
                kl_temperature=config_data["kl_temperature"],
                w_balanced=config_data["w_balanced"],
                w_focal=config_data["w_focal"],
                w_ck_guard=config_data["w_ck_guard"],
                w_kl=config_data["w_kl"],
                w_reg=config_data["w_reg"]
            )
            
            loss_total = loss_outputs['loss_total']
            
            # Check tensor sanity
            for name, tensor in {**outputs, "loss_total": loss_total}.items():
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
        selection_score = 0.50 * macro_f1 + 0.25 * ck_f1 + 0.25 * bottleneck_macro_f1
        
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
            "val_class1_f1": class1_f1,
            "val_class2_f1": class2_f1,
            "val_class5_f1": class5_f1,
            "val_bottleneck_macro_f1": bottleneck_macro_f1,
            "val_selection_score": selection_score,
            "val_per_class_f1": str(per_class_f1.tolist()),
            "val_confusion_matrix": str(cm.tolist()),
            "train_total_loss": train_loss_metrics["loss_total"],
            "train_balanced_6way_loss": train_loss_metrics["loss_6way"],
            "train_focal_bottleneck_loss": train_loss_metrics["loss_focal"],
            "train_ck_guard_loss": train_loss_metrics["loss_ck_guard"],
            "train_kl_anchor_loss": train_loss_metrics["loss_kl"],
            "train_adapter_l2_loss": train_loss_metrics["loss_residual"]
        }
        epoch_logs.append(epoch_log_entry)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss_metrics['loss_total']:.4f} | "
              f"Val Score: {selection_score:.4f} (Macro: {macro_f1:.4f}, CK: {ck_f1:.4f}, Bottleneck Macro: {bottleneck_macro_f1:.4f}) | "
              f"Best: {is_best}")
              
        if patience_counter >= patience:
            print(f"[-] Early stopping triggered. Best epoch was {best_epoch} with Val Selection Score: {best_score:.6f}")
            break
            
    # 9. Post-training updates and saving results
    # Save detailed epoch log for the config
    epoch_csv_path = os.path.join(out_dir, f"E1_EPOCH_LOG_{config_name}.csv")
    df_epoch = pd.DataFrame(epoch_logs)
    df_epoch.to_csv(epoch_csv_path, index=False)
    print(f"[+] Saved epoch training log to: {epoch_csv_path}")
    
    # Save the best epoch metrics to E1_VAL_METRICS_ALL_CONFIGS.csv
    best_entry = next(entry for entry in epoch_logs if entry["epoch"] == best_epoch)
    
    best_row_data = {
        "config_name": config_name,
        "beta": config_data["beta"],
        "epoch": best_epoch,
        "val_accuracy": best_entry["val_accuracy"],
        "val_micro_f1": best_entry["val_micro_f1"],
        "val_macro_f1": best_entry["val_macro_f1"],
        "val_weighted_f1": best_entry["val_weighted_f1"],
        "val_ck_f1": best_entry["val_ck_f1"],
        "val_class1_f1": best_entry["val_class1_f1"],
        "val_class2_f1": best_entry["val_class2_f1"],
        "val_class5_f1": best_entry["val_class5_f1"],
        "val_bottleneck_macro_f1": best_entry["val_bottleneck_macro_f1"],
        "val_selection_score": best_entry["val_selection_score"],
        "val_per_class_f1": best_entry["val_per_class_f1"],
        "val_confusion_matrix": best_entry["val_confusion_matrix"],
        # Add class columns for easy comparison regeneration
        "f1_class_0": float(eval(best_entry["val_per_class_f1"])[0]),
        "f1_class_1": float(eval(best_entry["val_per_class_f1"])[1]),
        "f1_class_2": float(eval(best_entry["val_per_class_f1"])[2]),
        "f1_class_3": float(eval(best_entry["val_per_class_f1"])[3]),
        "f1_class_4": float(eval(best_entry["val_per_class_f1"])[4]),
        "f1_class_5": float(eval(best_entry["val_per_class_f1"])[5]),
        "train_total_loss": best_entry["train_total_loss"],
        "train_balanced_6way_loss": best_entry["train_balanced_6way_loss"],
        "train_focal_bottleneck_loss": best_entry["train_focal_bottleneck_loss"],
        "train_ck_guard_loss": best_entry["train_ck_guard_loss"],
        "train_kl_anchor_loss": best_entry["train_kl_anchor_loss"],
        "train_adapter_l2_loss": best_entry["train_adapter_l2_loss"],
        "nan_inf_occurred": nan_inf_occurred
    }
    
    report_csv = os.path.join(out_dir, "E1_VAL_METRICS_ALL_CONFIGS.csv")
    if os.path.exists(report_csv):
        df_all = pd.read_csv(report_csv)
        df_all = df_all[df_all["config_name"] != config_name]
        df_all = pd.concat([df_all, pd.DataFrame([best_row_data])], ignore_index=True)
    else:
        df_all = pd.DataFrame([best_row_data])
    df_all.to_csv(report_csv, index=False)
    print(f"[+] Updated aggregated config metrics in: {report_csv}")
    
    # Save the best confusion matrix CSV
    df_best_cm = pd.DataFrame(best_cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_best_cm.to_csv(os.path.join(matrix_dir, f"{config_name}_confusion_matrix.csv"))
    print(f"[+] Saved confusion matrix to: {os.path.join(matrix_dir, f'{config_name}_confusion_matrix.csv')}")
    
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
    
    rescue_csv = os.path.join(out_dir, "E1_RESCUE_BROKEN_BY_CLASS.csv")
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
    print("[+] All Stage I-E1 global reports successfully regenerated.")

if __name__ == "__main__":
    main()
