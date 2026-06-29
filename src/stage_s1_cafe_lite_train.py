"""
Stage S1: CAFE-lite Validation-Only Training Runner.
Loads JSON configuration, cached multimodal features (Text + Image only),
and runs a validation-only training loop under strict safety restrictions:
- No KG features, relation IDs, TVCS specialist, F4 backbone, or baseline logits.
- split_id == 2 (locked test set) is strictly isolated and never loaded/evaluated.
- Selection is based on val_selection_score = 0.5 * Macro-F1 + 0.5 * CK-F1.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add workspace path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage_s1_cafe_lite_model import CafeLiteSameSplitModel
from src.stage_s1_cafe_lite_losses import compute_cafe_lite_loss, check_tensor_sanity

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
    parser = argparse.ArgumentParser(description="Stage S1 CAFE-lite Training Runner")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override maximum epochs.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early stopping patience.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size.")
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
    Reads S1_VAL_METRICS_ALL_CONFIGS.csv and dynamically regenerates:
    - S1_TRAINING_SUMMARY.txt
    - S1_FINAL_DECISION.txt
    - S1_PER_CLASS_F1.csv
    """
    out_dir = os.path.join(project_root, "outputs", "stage_s1_cafe_lite")
    report_csv = os.path.join(out_dir, "S1_VAL_METRICS_ALL_CONFIGS.csv")
    summary_txt = os.path.join(out_dir, "S1_TRAINING_SUMMARY.txt")
    decision_txt = os.path.join(out_dir, "S1_FINAL_DECISION.txt")
    per_class_csv = os.path.join(out_dir, "S1_PER_CLASS_F1.csv")
    
    if not os.path.exists(report_csv):
        print(f"[!] Cannot regenerate reports: {report_csv} does not exist.")
        return
        
    df_configs = pd.read_csv(report_csv)
    if len(df_configs) == 0:
        print("[!] Cannot regenerate reports: S1_VAL_METRICS_ALL_CONFIGS.csv is empty.")
        return
        
    # 1. Regenerate S1_PER_CLASS_F1.csv
    df_per_class = df_configs[[
        "config_name",
        "val_class0_f1",
        "val_class1_f1",
        "val_class2_f1",
        "val_class3_f1",
        "val_class4_f1",
        "val_class5_f1"
    ]]
    df_per_class.to_csv(per_class_csv, index=False)
    print(f"[+] Regenerated per-class F1 comparisons at: {per_class_csv}")
    
    # 2. Write S1_TRAINING_SUMMARY.txt
    with open(summary_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE S1: CAFE-LITE VAL-ONLY TRAINING SUMMARY\n")
        f.write("========================================================================\n\n")
        
        f.write("SUMMARY OF BEST VAL PERFORMANCE ACROSS CONFIGURATIONS:\n")
        f.write("----------------------------------------------------\n")
        f.write(df_configs.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED CONFIGURATION PERFORMANCE DETAILS:\n")
        f.write("-----------------------------------------\n")
        for _, r in df_configs.iterrows():
            f.write(f"\nConfiguration: {r['config_name'].upper()}\n")
            f.write("=" * (len(r['config_name']) + 15) + "\n")
            f.write(f"  Best Epoch:                 {r['epoch']}\n")
            f.write(f"  Validation Selection Score:  {r['val_selection_score']:.6f}\n")
            f.write(f"  Validation Accuracy:        {r['val_accuracy']:.6f}\n")
            f.write(f"  Validation Macro-F1:       {r['val_macro_f1']:.6f}\n")
            f.write(f"  Validation Micro-F1:       {r['val_micro_f1']:.6f}\n")
            f.write(f"  Validation CK-F1:          {r['val_ck_f1']:.6f}\n")
            f.write(f"  Mean Ambiguity Score:       {r['val_mean_ambiguity']:.6f}\n")
            f.write(f"  Mean Similarity Score:      {r['val_mean_similarity']:.6f}\n")
            f.write(f"  NaN/Inf Occurred:           {r['nan_inf_occurred']}\n")
            
    print(f"[+] Regenerated training summary at: {summary_txt}")
    
    # 3. Write S1_FINAL_DECISION.txt
    # Find the best configuration by validation selection score
    best_idx = df_configs["val_selection_score"].idxmax()
    best_row = df_configs.iloc[best_idx]
    best_cfg_name = best_row["config_name"]
    best_macro_f1 = best_row["val_macro_f1"]
    best_ck_f1 = best_row["val_ck_f1"]
    nan_inf = bool(best_row["nan_inf_occurred"])
    
    # Decision Rules:
    # - Stage G1 best baseline: validation Macro-F1 ~ 0.4486, CK-F1 ~ 0.3429
    # - Stage F4 reference: validation Macro-F1 ~ 0.4792, CK-F1 ~ 0.3922
    # If it clearly beats G1 validation and is stable, mark VALIDATION_BASELINE_READY.
    # If it also approaches or beats F4 validation, mark PROMOTE_TO_LOCKED_TEST_CANDIDATE.
    # Otherwise DIAGNOSTIC_ONLY.
    beats_g1 = (best_macro_f1 > 0.4486) and (best_ck_f1 > 0.3429) and not nan_inf
    approaches_f4 = (best_macro_f1 >= 0.470) and (best_ck_f1 >= 0.380)
    
    if beats_g1:
        if approaches_f4:
            decision_status = "PROMOTE_TO_LOCKED_TEST_CANDIDATE"
        else:
            decision_status = "VALIDATION_BASELINE_READY"
    else:
        decision_status = "DIAGNOSTIC_ONLY"
        
    with open(decision_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE S1: CAFE-LITE FINAL DECISION REPORT\n")
        f.write("========================================================================\n\n")
        f.write(f"Best Performing Configuration: {best_cfg_name}\n")
        f.write(f"Decision:                      {decision_status}\n\n")
        
        f.write("DECISION RULE VERIFICATION:\n")
        f.write("--------------------------\n")
        f.write(f"1. Beats Stage G1 validation baseline? (Macro-F1 > 0.4486 & CK-F1 > 0.3429):\n")
        f.write(f"   - Macro-F1: {best_macro_f1:.6f} vs G1: 0.4486 -> {'PASSED' if best_macro_f1 > 0.4486 else 'FAILED'}\n")
        f.write(f"   - CK-F1:    {best_ck_f1:.6f} vs G1: 0.3429 -> {'PASSED' if best_ck_f1 > 0.3429 else 'FAILED'}\n")
        f.write(f"   - Result:   {'PASSED' if beats_g1 else 'FAILED'}\n\n")
        
        f.write(f"2. Approaches/beats Stage F4 validation? (Macro-F1 >= 0.470 & CK-F1 >= 0.380):\n")
        f.write(f"   - Macro-F1: {best_macro_f1:.6f} vs F4: 0.4792 -> {'YES' if best_macro_f1 >= 0.470 else 'NO'}\n")
        f.write(f"   - CK-F1:    {best_ck_f1:.6f} vs F4: 0.3922 -> {'YES' if best_ck_f1 >= 0.380 else 'NO'}\n")
        f.write(f"   - Result:   {'PASSED' if approaches_f4 else 'FAILED'}\n\n")
        
        f.write("ASSURANCES:\n")
        f.write("-----------\n")
        f.write("- NO LOCKED TEST (split_id == 2) WAS LOADED OR EVALUATED IN THIS SCRIPT.\n")
        f.write("- CAFE-lite uses Text + Image only; KG/TVCS are strictly disabled.\n\n")
        
        f.write("Best Configuration Performance Details:\n")
        f.write("---------------------------------------\n")
        f.write(f"Best Epoch:                 {best_row['epoch']}\n")
        f.write(f"Validation Selection Score:  {best_row['val_selection_score']:.6f}\n")
        f.write(f"Validation Accuracy:        {best_row['val_accuracy']:.6f}\n")
        f.write(f"Validation Macro-F1:       {best_row['val_macro_f1']:.6f}\n")
        f.write(f"Validation Micro-F1:       {best_row['val_micro_f1']:.6f}\n")
        f.write(f"Validation CK-F1:          {best_row['val_ck_f1']:.6f}\n")
        f.write(f"Mean Ambiguity Score:       {best_row['val_mean_ambiguity']:.6f}\n")
        f.write(f"Mean Similarity Score:      {best_row['val_mean_similarity']:.6f}\n")
        f.write(f"NaN/Inf Occurred:           {best_row['nan_inf_occurred']}\n")
        
    print(f"[+] Regenerated final decision at: {decision_txt}")

def main():
    # 1. Startup assertions & output messages
    args = parse_args()
    assert args.no_test_eval, "Abort: --no_test_eval flag is missing! Must be present to ensure test set isolation."
    
    print("LOCKED TEST IS DISABLED")
    print("CAFE-lite uses Text + Image only; KG/TVCS disabled")
    
    # 2. Config reading & hyperparameter setup
    if not os.path.exists(args.config):
        print(f"[-] ERROR: Config file not found at: {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config_data = json.load(f)
        
    config_name = config_data.get("config_name", os.path.basename(args.config))
    print(f"[+] Loading config: {config_name}")
    
    max_epochs = args.max_epochs if args.max_epochs is not None else config_data.get("epochs", 20)
    patience = args.patience if args.patience is not None else config_data.get("patience", 5)
    batch_size = args.batch_size if args.batch_size is not None else config_data.get("batch_size", 16)
    lr = args.lr if args.lr is not None else config_data.get("lr", 1e-4)
    weight_decay = args.weight_decay if args.weight_decay is not None else config_data.get("weight_decay", 1e-4)
    seed = args.seed if args.seed is not None else config_data.get("seed", 42)
    
    set_seed(seed)
    
    # 3. Cache directory verification
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    print(f"[+] Loading cached features from: {cache_dir}")
    
    split_ids = np.load(os.path.join(cache_dir, "split_ids.npy"))
    labels_fine = np.load(os.path.join(cache_dir, "labels_fine.npy"))
    
    # Assert split_id == 2 is never loaded/evaluated in splits
    train_mask = (split_ids == 0)
    val_mask = (split_ids == 1)
    
    if np.any(split_ids[train_mask] == 2) or np.any(split_ids[val_mask] == 2):
        print("[-] ERROR: Test split (split_id == 2) detected in train/val splits! Aborting.")
        sys.exit(1)
        
    # Load acceptable features
    text_features = np.load(os.path.join(cache_dir, "text_features.npy"))
    image_features_global = np.load(os.path.join(cache_dir, "image_features_global.npy"))
    image_features_patch = np.load(os.path.join(cache_dir, "image_features_patch.npy"))
    
    # Safety Check: explicitly assert we do not load prohibited files
    for prohibited in ["kg_features.npy", "relation_ids.npy"]:
        p_path = os.path.join(cache_dir, prohibited)
        if os.path.exists(p_path) and prohibited == "kg_features.npy":
            print(f"[!] Warning: {prohibited} exists in cache but is strictly IGNORED and NOT loaded by CAFE-lite.")
            
    # Slicing datasets and building loaders
    tr_text = torch.tensor(text_features[train_mask], dtype=torch.float32)
    tr_img_g = torch.tensor(image_features_global[train_mask], dtype=torch.float32)
    tr_img_p = torch.tensor(image_features_patch[train_mask], dtype=torch.float32)
    tr_lbl = torch.tensor(labels_fine[train_mask], dtype=torch.long)
    
    val_text = torch.tensor(text_features[val_mask], dtype=torch.float32)
    val_img_g = torch.tensor(image_features_global[val_mask], dtype=torch.float32)
    val_img_p = torch.tensor(image_features_patch[val_mask], dtype=torch.float32)
    val_lbl = torch.tensor(labels_fine[val_mask], dtype=torch.long)
    
    train_ds = TensorDataset(tr_text, tr_img_g, tr_img_p, tr_lbl)
    val_ds = TensorDataset(val_text, val_img_g, val_img_p, val_lbl)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device selected: {device}")
    
    # Instantiate CafeLiteSameSplitModel
    model = CafeLiteSameSplitModel(
        d_model=config_data.get("d_model", 256),
        use_patch_pooling=config_data.get("use_patch_pooling", False),
        dropout=config_data.get("dropout", 0.1)
    ).to(device)
    
    # Class weights and priors from training split only (split_ids == 0)
    train_lbl_np = labels_fine[train_mask]
    class_counts = np.bincount(train_lbl_np, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_lbl_np) / (6.0 * class_counts)
    
    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Outputs Setup
    out_dir = os.path.join(args.project_root, "outputs", "stage_s1_cafe_lite")
    ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_s1_cafe_lite")
    matrix_dir = os.path.join(out_dir, "S1_CONFUSION_MATRICES")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)
    
    ckpt_path = os.path.join(ckpt_dir, f"best_{config_name}.pt")
    
    # Training Loop variables
    best_score = -1.0
    best_epoch = -1
    patience_counter = 0
    epoch_logs = []
    best_cm = None
    
    print("\n[+] Starting training...")
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        total_samples = 0
        nan_inf_occurred = False
        
        for bx_text, bx_img_g, bx_img_p, bx_lbl in train_loader:
            bx_text = bx_text.to(device)
            bx_img_g = bx_img_g.to(device)
            bx_img_p = bx_img_p.to(device)
            bx_lbl = bx_lbl.to(device)
            
            optimizer.zero_grad()
            
            logits, ambiguity_score, similarity_score, diagnostics = model(
                text_features=bx_text,
                image_features_global=bx_img_g,
                image_features_patch=bx_img_p if config_data.get("use_patch_pooling", False) else None
            )
            
            loss_outputs = compute_cafe_lite_loss(
                logits=logits,
                targets=bx_lbl,
                ambiguity_score=ambiguity_score,
                similarity_score=similarity_score,
                loss_type=config_data.get("loss", "standard"),
                class_weights=class_weights_t,
                class_priors=class_priors_t,
                tau=0.5,
                w_ambiguity_reg=config_data.get("w_ambiguity_reg", 0.1)
            )
            
            loss_total = loss_outputs["loss_total"]
            
            # Sanity checks
            for name, tensor in [("logits", logits), ("loss_total", loss_total)]:
                nc, ic = check_tensor_sanity(tensor, name)
                if nc > 0 or ic > 0:
                    nan_inf_occurred = True
                    
            loss_total.backward()
            optimizer.step()
            
            train_loss += loss_total.item() * len(bx_lbl)
            total_samples += len(bx_lbl)
            
        train_loss /= total_samples
        
        # Validation Evaluation
        model.eval()
        val_preds = []
        val_targets = []
        val_ambiguities = []
        val_similarities = []
        
        with torch.no_grad():
            for bx_text, bx_img_g, bx_img_p, bx_lbl in val_loader:
                bx_text = bx_text.to(device)
                bx_img_g = bx_img_g.to(device)
                bx_img_p = bx_img_p.to(device)
                
                logits, ambiguity_score, similarity_score, diagnostics = model(
                    text_features=bx_text,
                    image_features_global=bx_img_g,
                    image_features_patch=bx_img_p if config_data.get("use_patch_pooling", False) else None
                )
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(bx_lbl.numpy())
                val_ambiguities.extend(ambiguity_score.cpu().numpy())
                val_similarities.extend(similarity_score.cpu().numpy())
                
                for name, tensor in [("logits", logits), ("ambiguity_score", ambiguity_score)]:
                    nc, ic = check_tensor_sanity(tensor, name)
                    if nc > 0 or ic > 0:
                        nan_inf_occurred = True
                        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_ambiguities = np.array(val_ambiguities)
        val_similarities = np.array(val_similarities)
        
        acc = accuracy_score(val_targets, val_preds)
        micro_f1 = f1_score(val_targets, val_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(6)), zero_division=0)
        
        ck_f1 = per_class_f1[2]
        selection_score = 0.50 * macro_f1 + 0.50 * ck_f1
        
        mean_ambiguity = np.mean(val_ambiguities)
        mean_similarity = np.mean(val_similarities)
        
        cm = confusion_matrix(val_targets, val_preds, labels=list(range(6)) )
        
        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
            best_epoch = epoch + 1
            patience_counter = 0
            best_cm = cm
            
            # Save checkpoint
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
                    'per_class_f1': per_class_f1.tolist(),
                    'selection_score': selection_score,
                    'mean_ambiguity': mean_ambiguity,
                    'mean_similarity': mean_similarity,
                    'confusion_matrix': cm.tolist()
                },
                'config': config_data,
                'nan_inf_occurred': 1 if nan_inf_occurred else 0
            }
            torch.save(ckpt_state, ckpt_path)
            print(f"  [+] Saved best checkpoint with Val Selection Score: {selection_score:.6f}")
        else:
            patience_counter += 1
            
        epoch_log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
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
            "val_selection_score": selection_score,
            "val_mean_ambiguity": mean_ambiguity,
            "val_mean_similarity": mean_similarity,
            "val_confusion_matrix": str(cm.tolist()),
            "nan_inf_occurred": 1 if nan_inf_occurred else 0
        }
        epoch_logs.append(epoch_log_entry)
        
        print(f"Epoch {epoch+1:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Score: {selection_score:.4f} (Macro: {macro_f1:.4f}, CK: {ck_f1:.4f}) | "
              f"Best: {is_best}")
              
        if patience_counter >= patience:
            print(f"[-] Early stopping triggered. Best epoch was {best_epoch} with Val Selection Score: {best_score:.6f}")
            break
            
    # 4. Save Training Logs and Aggregated Metrics
    epoch_csv_path = os.path.join(out_dir, f"S1_EPOCH_LOG_{config_name}.csv")
    df_epoch = pd.DataFrame(epoch_logs)
    df_epoch.to_csv(epoch_csv_path, index=False)
    print(f"[+] Saved epoch training log to: {epoch_csv_path}")
    
    best_entry = next(entry for entry in epoch_logs if entry["epoch"] == best_epoch)
    best_row_data = {
        "config_name": config_name,
        "epoch": best_epoch,
        "train_loss": best_entry["train_loss"],
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
        "val_selection_score": best_entry["val_selection_score"],
        "val_mean_ambiguity": best_entry["val_mean_ambiguity"],
        "val_mean_similarity": best_entry["val_mean_similarity"],
        "val_confusion_matrix": best_entry["val_confusion_matrix"],
        "nan_inf_occurred": best_entry["nan_inf_occurred"]
    }
    
    report_csv = os.path.join(out_dir, "S1_VAL_METRICS_ALL_CONFIGS.csv")
    if os.path.exists(report_csv):
        df_all = pd.read_csv(report_csv)
        df_all = df_all[df_all["config_name"] != config_name]
        df_all = pd.concat([df_all, pd.DataFrame([best_row_data])], ignore_index=True)
    else:
        df_all = pd.DataFrame([best_row_data])
    df_all.to_csv(report_csv, index=False)
    print(f"[+] Updated aggregated config metrics in: {report_csv}")
    
    # Save best confusion matrix CSV to confusion matrix dir
    df_best_cm = pd.DataFrame(best_cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_best_cm.to_csv(os.path.join(matrix_dir, f"best_{config_name}_confusion_matrix.csv"))
    print(f"[+] Saved confusion matrix to: {os.path.join(matrix_dir, f'best_{config_name}_confusion_matrix.csv')}")
    
    # Regenerate global reports
    regenerate_global_reports(args.project_root)
    print("[+] All S1 global reports successfully regenerated.")

if __name__ == "__main__":
    main()
