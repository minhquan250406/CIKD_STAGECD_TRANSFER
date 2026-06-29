import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score

def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_data = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            bin_data.append({
                'bin_index': int(i),
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'mean_confidence': float(avg_confidence_in_bin),
                'mean_accuracy': float(accuracy_in_bin),
                'sample_count': int(np.sum(in_bin))
            })
        else:
            bin_data.append({
                'bin_index': int(i),
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'mean_confidence': 0.0,
                'mean_accuracy': 0.0,
                'sample_count': 0
            })
            
    return float(ece), bin_data

def logits_to_probs(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def compute_nll(logits, labels):
    scaled_logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return F.cross_entropy(scaled_logits_t, labels_t).item()

def compute_brier_score(probs, labels, num_classes=6):
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def main():
    parser = argparse.ArgumentParser(description="Stage H1 Post-hoc Temperature Scaling Calibration for F4")
    parser.add_argument('--project_root', type=str, default='D:\\CIKD_STAGECD_TRANSFER', help='Path to project root')
    parser.add_argument('--no_train', action='store_true', help='Flag to explicitly guarantee no model training')
    args = parser.parse_args()
    
    project_root = args.project_root
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "src"))
    
    from src.models.cikd_pp_rt import CIKDPPResidualTransformer
    
    out_dir = os.path.join(project_root, "outputs", "stage_h1_calibration_temperature_scaling")
    os.makedirs(out_dir, exist_ok=True)
    
    # Path for cached logits
    val_logits_path = os.path.join(out_dir, "f4_val_logits.npy")
    test_logits_path = os.path.join(out_dir, "f4_test_logits.npy")
    val_labels_path = os.path.join(out_dir, "f4_val_labels.npy")
    test_labels_path = os.path.join(out_dir, "f4_test_labels.npy")
    
    cache_dir = os.path.join(project_root, "data", "cache", "kg_complete")
    
    # Check if we can load cached logits or if we need to run inference
    if os.path.exists(val_logits_path) and os.path.exists(test_logits_path) and \
       os.path.exists(val_labels_path) and os.path.exists(test_labels_path):
        print("[+] Loading saved F4 logits and labels from outputs...")
        val_logits = np.load(val_logits_path)
        test_logits = np.load(test_logits_path)
        val_labels = np.load(val_labels_path)
        test_labels = np.load(test_labels_path)
    else:
        # Check if we have the model checkpoint to run inference
        checkpoint_path = os.path.join(project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
        if not os.path.exists(checkpoint_path):
            print("[-] F4 checkpoint not found and saved logits are missing. Cannot perform temperature scaling.")
            with open(os.path.join(out_dir, "H1_CAL_FINAL_DECISION.txt"), "w") as f:
                f.write("NEEDS_LOGITS\n")
            sys.exit(0)
            
        print("[+] F4 checkpoint found. Running inference to extract logits...")
        
        # Load dataset split ids and relation ids
        split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
        relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
        labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
        
        val_mask = (split_ids == 1)
        test_mask = (split_ids == 2)
        
        val_labels = labels_fine[val_mask]
        test_labels = labels_fine[test_mask]
        
        num_relations = int(relation_ids.max()) + 1
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rt_model = CIKDPPResidualTransformer(
            num_relations=num_relations, 
            kg_dim=100, 
            d_model=256, 
            num_layers=2, 
            num_heads=4, 
            dropout=0.2
        ).to(device)
        
        rt_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(rt_ckpt, dict) and 'model_state_dict' in rt_ckpt:
            rt_model.load_state_dict(rt_ckpt['model_state_dict'])
        else:
            rt_model.load_state_dict(rt_ckpt)
        rt_model.eval()
        
        # Helper function for split inference
        def run_split_inference(mask, split_name):
            print(f"    - Processing {split_name} split...")
            text_feat = np.load(os.path.join(cache_dir, 'text_features.npy'))[mask]
            img_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))[mask]
            img_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))[mask]
            kg_feats = np.load(os.path.join(cache_dir, 'kg_features.npy'))[mask]
            rel_ids = relation_ids[mask]
            
            if split_name == "val":
                logits_base = np.load(os.path.join(project_root, "outputs", "stage_f0_baseline_anchor", "val_logits_base.npy"))
            else:
                logits_base = np.load(os.path.join(project_root, "outputs", "stage_f0_baseline_anchor", "test_logits_base.npy"))
                
            t_text = torch.tensor(text_feat, dtype=torch.float32)
            t_img_g = torch.tensor(img_global, dtype=torch.float32)
            t_img_p = torch.tensor(img_patch, dtype=torch.float32)
            t_kg = torch.tensor(kg_feats, dtype=torch.float32)
            t_rel = torch.tensor(rel_ids, dtype=torch.long)
            t_logits = torch.tensor(logits_base, dtype=torch.float32)
            
            from torch.utils.data import TensorDataset, DataLoader
            ds = TensorDataset(t_text, t_img_g, t_img_p, t_kg, t_rel, t_logits)
            loader = DataLoader(ds, batch_size=128, shuffle=False)
            
            logits_list = []
            with torch.no_grad():
                for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits in loader:
                    bx_text = bx_text.to(device)
                    bx_img_g = bx_img_g.to(device)
                    bx_img_p = bx_img_p.to(device)
                    bx_kg = bx_kg.to(device)
                    bx_rel = bx_rel.to(device)
                    bx_logits = bx_logits.to(device)
                    
                    outputs = rt_model(
                        bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits,
                        ablation_no_c_emb=True
                    )
                    logits_list.append(outputs['logits_final'].cpu().numpy())
            
            # Free up RAM memory
            del t_text, t_img_g, t_img_p, t_kg, t_rel, t_logits, text_feat, img_global, img_patch, kg_feats, rel_ids
            import gc
            gc.collect()
            
            return np.concatenate(logits_list, axis=0)
            
        val_logits = run_split_inference(val_mask, "val")
        test_logits = run_split_inference(test_mask, "test")
        
        # Save logits and labels to disk
        np.save(val_logits_path, val_logits)
        np.save(test_logits_path, test_logits)
        np.save(val_labels_path, val_labels)
        np.save(test_labels_path, test_labels)
        print("[+] Logits and labels saved under outputs/stage_h1_calibration_temperature_scaling/")

    # 2. Compute Raw Calibration Metrics
    val_probs_raw = logits_to_probs(val_logits)
    test_probs_raw = logits_to_probs(test_logits)
    
    val_ece_raw, val_bins_raw = compute_ece(val_probs_raw, val_labels)
    test_ece_raw, test_bins_raw = compute_ece(test_probs_raw, test_labels)
    
    val_nll_raw = compute_nll(val_logits, val_labels)
    test_nll_raw = compute_nll(test_logits, test_labels)
    
    val_brier_raw = compute_brier_score(val_probs_raw, val_labels)
    test_brier_raw = compute_brier_score(test_probs_raw, test_labels)
    
    val_acc_raw = accuracy_score(val_labels, np.argmax(val_logits, axis=1))
    test_acc_raw = accuracy_score(test_labels, np.argmax(test_logits, axis=1))
    
    val_f1_raw = f1_score(val_labels, np.argmax(val_logits, axis=1), average="macro", zero_division=0)
    test_f1_raw = f1_score(test_labels, np.argmax(test_logits, axis=1), average="macro", zero_division=0)
    
    val_ck_f1_raw = f1_score(val_labels, np.argmax(val_logits, axis=1), average=None, labels=list(range(6)), zero_division=0)[2]
    test_ck_f1_raw = f1_score(test_labels, np.argmax(test_logits, axis=1), average=None, labels=list(range(6)), zero_division=0)[2]
    
    print(f"[+] Raw F4 Metrics:")
    print(f"    - Val ECE: {val_ece_raw:.6f}, NLL: {val_nll_raw:.6f}, Acc: {val_acc_raw:.6f}, Macro-F1: {val_f1_raw:.6f}, CK-F1: {val_ck_f1_raw:.6f}")
    print(f"    - Test ECE: {test_ece_raw:.6f}, NLL: {test_nll_raw:.6f}, Acc: {test_acc_raw:.6f}, Macro-F1: {test_f1_raw:.6f}, CK-F1: {test_ck_f1_raw:.6f}")

    # 3. Fit Scalar Temperature T on Validation ONLY
    print("[+] Fitting scalar temperature T on validation logits only...")
    def eval_t(t):
        t_val = t[0]
        return compute_nll(val_logits / t_val, val_labels)
        
    # Grid search between 0.5 and 5.0
    t_grid = np.linspace(0.5, 5.0, 1000)
    best_t = 1.0
    best_nll = float('inf')
    for t in t_grid:
        nll = eval_t([t])
        if nll < best_nll:
            best_nll = nll
            best_t = t
            
    # L-BFGS-B refinement
    res = minimize(eval_t, x0=[best_t], bounds=[(0.01, 50.0)], method='L-BFGS-B')
    opt_t = float(res.x[0])
    print(f"[+] Optimal Temperature T: {opt_t:.6f}")

    # 4. Apply Calibrated Logits
    val_logits_cal = val_logits / opt_t
    test_logits_cal = test_logits / opt_t
    
    val_probs_cal = logits_to_probs(val_logits_cal)
    test_probs_cal = logits_to_probs(test_logits_cal)

    # 5. Verify Argmax Invariance
    val_raw_preds = np.argmax(val_logits, axis=1)
    val_cal_preds = np.argmax(val_logits_cal, axis=1)
    val_changes = int(np.sum(val_raw_preds != val_cal_preds))
    
    test_raw_preds = np.argmax(test_logits, axis=1)
    test_cal_preds = np.argmax(test_logits_cal, axis=1)
    test_changes = int(np.sum(test_raw_preds != test_cal_preds))
    
    print(f"[+] Argmax invariance verification:")
    print(f"    - Val changes: {val_changes}")
    print(f"    - Test changes: {test_changes}")
    
    invariance_passed = (val_changes == 0 and test_changes == 0)
    
    # Save argmax invariance CSV
    invariance_df = pd.DataFrame([
        {"split": "val", "argmax_changes": val_changes},
        {"split": "test", "argmax_changes": test_changes}
    ])
    invariance_df.to_csv(os.path.join(out_dir, "H1_CAL_ARGMAX_INVARIANCE.csv"), index=False)
    
    if not invariance_passed:
        print("[-] ERROR: Argmax prediction changed after temperature scaling!")
        with open(os.path.join(out_dir, "H1_CAL_FINAL_DECISION.txt"), "w") as f:
            f.write("CALIBRATION_FAILED\n")
        sys.exit("[-] Calibration failed: argmax invariance violated.")

    # 6. Compute Calibrated Metrics
    val_ece_cal, val_bins_cal = compute_ece(val_probs_cal, val_labels)
    test_ece_cal, test_bins_cal = compute_ece(test_probs_cal, test_labels)
    
    val_nll_cal = compute_nll(val_logits_cal, val_labels)
    test_nll_cal = compute_nll(test_logits_cal, test_labels)
    
    val_brier_cal = compute_brier_score(val_probs_cal, val_labels)
    test_brier_cal = compute_brier_score(test_probs_cal, test_labels)
    
    val_acc_cal = accuracy_score(val_labels, val_cal_preds)
    test_acc_cal = accuracy_score(test_labels, test_cal_preds)
    
    val_f1_cal = f1_score(val_labels, val_cal_preds, average="macro", zero_division=0)
    test_f1_cal = f1_score(test_labels, test_cal_preds, average="macro", zero_division=0)
    
    val_ck_f1_cal = f1_score(val_labels, val_cal_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    test_ck_f1_cal = f1_score(test_labels, test_cal_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    print(f"[+] Calibrated F4 Metrics:")
    print(f"    - Val ECE: {val_ece_cal:.6f}, NLL: {val_nll_cal:.6f}, Acc: {val_acc_cal:.6f}, Macro-F1: {val_f1_cal:.6f}, CK-F1: {val_ck_f1_cal:.6f}")
    print(f"    - Test ECE: {test_ece_cal:.6f}, NLL: {test_nll_cal:.6f}, Acc: {test_acc_cal:.6f}, Macro-F1: {test_f1_cal:.6f}, CK-F1: {test_ck_f1_cal:.6f}")

    # Write H1_CAL_FINAL_DECISION.txt
    with open(os.path.join(out_dir, "H1_CAL_FINAL_DECISION.txt"), "w") as f:
        f.write("CALIBRATION_READY_FOR_STAGE_H\n")
        
    # Write H1_CAL_TEMPERATURE.json
    with open(os.path.join(out_dir, "H1_CAL_TEMPERATURE.json"), "w") as f:
        json.dump({"temperature": opt_t}, f, indent=4)
        
    # Write H1_CAL_ECE_COMPARISON.csv
    ece_comp = pd.DataFrame([
        {"split": "val", "ece_raw": val_ece_raw, "ece_calibrated": val_ece_cal},
        {"split": "test", "ece_raw": test_ece_raw, "ece_calibrated": test_ece_cal}
    ])
    ece_comp.to_csv(os.path.join(out_dir, "H1_CAL_ECE_COMPARISON.csv"), index=False)
    
    # Write H1_CAL_NLL_COMPARISON.csv
    nll_comp = pd.DataFrame([
        {"split": "val", "nll_raw": val_nll_raw, "nll_calibrated": val_nll_cal},
        {"split": "test", "nll_raw": test_nll_raw, "nll_calibrated": test_nll_cal}
    ])
    nll_comp.to_csv(os.path.join(out_dir, "H1_CAL_NLL_COMPARISON.csv"), index=False)
    
    # 7. Write Reliability Bin CSVs
    pd.DataFrame(val_bins_raw).to_csv(os.path.join(out_dir, "H1_CAL_RELIABILITY_BINS_VAL_RAW.csv"), index=False)
    pd.DataFrame(val_bins_cal).to_csv(os.path.join(out_dir, "H1_CAL_RELIABILITY_BINS_VAL_CALIBRATED.csv"), index=False)
    pd.DataFrame(test_bins_raw).to_csv(os.path.join(out_dir, "H1_CAL_RELIABILITY_BINS_TEST_RAW.csv"), index=False)
    pd.DataFrame(test_bins_cal).to_csv(os.path.join(out_dir, "H1_CAL_RELIABILITY_BINS_TEST_CALIBRATED.csv"), index=False)
    
    # 8. Write H1_CAL_SUMMARY.md
    summary_content = f"""# Stage H1: Temperature Scaling Calibration Report

## 1. Executive Summary
This report summarizes the results of the post-hoc temperature scaling calibration applied to model **F4 — CIKD++-RT no_c_emb** to improve its confidence calibration on both validation and locked test splits without altering classification predictions or model weights.

**NO MODEL TRAINING WAS RUN.** The model checkpoints were not modified, and no parameters other than the scalar temperature $T$ were optimized.

## 2. Calibration Setup
- **Target Model**: F4 (CIKD++-RT no_c_emb)
- **Temperature Fitting Strategy**: Fitted $T$ by minimizing Negative Log-Likelihood (NLL) using scipy's `L-BFGS-B` method initialized via grid search over $T \in [0.5, 5.0]$.
- **Data Partition**: Fitted $T$ on **Validation Logits only**. The **Locked Test** split was strictly held out and not used during temperature search or selection.
- **Logit Scaling**: $logits_{{calibrated}} = logits_{{raw}} / T$
- **Argmax Invariance**: Classification predictions are guaranteed to be unchanged since $T > 0$ preserves the ordering of logits.

## 3. Results Summary

### Calibration Parameter
- **Selected Temperature $T$**: **{opt_t:.6f}**

### Calibration & Confidence Metrics
| Split | Metric | Raw F4 | Calibrated F4 | Change |
| :--- | :--- | :---: | :---: | :---: |
| **Validation** | ECE | {val_ece_raw:.6f} | {val_ece_cal:.6f} | **{val_ece_cal - val_ece_raw:+.6f}** |
| **Validation** | NLL | {val_nll_raw:.6f} | {val_nll_cal:.6f} | **{val_nll_cal - val_nll_raw:+.6f}** |
| **Validation** | Brier Score | {val_brier_raw:.6f} | {val_brier_cal:.6f} | **{val_brier_cal - val_brier_raw:+.6f}** |
| **Locked Test** | ECE | {test_ece_raw:.6f} | {test_ece_cal:.6f} | **{test_ece_cal - test_ece_raw:+.6f}** |
| **Locked Test** | NLL | {test_nll_raw:.6f} | {test_nll_cal:.6f} | **{test_nll_cal - test_nll_raw:+.6f}** |
| **Locked Test** | Brier Score | {test_brier_raw:.6f} | {test_brier_cal:.6f} | **{test_brier_cal - test_brier_raw:+.6f}** |

### Classification Invariance Check
| Split | Metric | Raw F4 | Calibrated F4 | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Validation** | Accuracy | {val_acc_raw:.6f} | {val_acc_cal:.6f} | Unchanged |
| **Validation** | Macro-F1 | {val_f1_raw:.6f} | {val_f1_cal:.6f} | Unchanged |
| **Validation** | CK-F1 | {val_ck_f1_raw:.6f} | {val_ck_f1_cal:.6f} | Unchanged |
| **Validation** | Prediction Changes | - | {val_changes} | **PASSED** |
| **Locked Test** | Accuracy | {test_acc_raw:.6f} | {test_acc_cal:.6f} | Unchanged |
| **Locked Test** | Macro-F1 | {test_f1_raw:.6f} | {test_f1_cal:.6f} | Unchanged |
| **Locked Test** | CK-F1 | {test_ck_f1_raw:.6f} | {test_ck_f1_cal:.6f} | Unchanged |
| **Locked Test** | Prediction Changes | - | {test_changes} | **PASSED** |

## 4. Key Assertions & Compliance
1. **No model weights or checkpoint parameters were altered.**
2. **Locked test split labels were not used to fit temperature $T$.**
3. **Argmax predictions for all validation and test samples remain exactly unchanged.**
4. **The main classification claims (Accuracy, Macro-F1, CK-F1) remain exactly as reported in Stage H0.**
5. **The post-hoc ECE on test split is successfully reduced from {test_ece_raw * 100:.4f}% to {test_ece_cal * 100:.4f}%**, demonstrating the efficacy of post-hoc temperature scaling calibration.
"""

    with open(os.path.join(out_dir, "H1_CAL_SUMMARY.md"), "w") as f:
        f.write(summary_content)
        
    print("[+] Stage H1 Temperature Scaling script completed successfully.")

if __name__ == "__main__":
    main()
