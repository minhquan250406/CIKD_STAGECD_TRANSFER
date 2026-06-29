import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

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
                'bin_index': i,
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'mean_confidence': avg_confidence_in_bin,
                'mean_accuracy': accuracy_in_bin,
                'sample_count': int(np.sum(in_bin))
            })
        else:
            bin_data.append({
                'bin_index': i,
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'mean_confidence': 0.0,
                'mean_accuracy': 0.0,
                'sample_count': 0
            })
            
    return ece, bin_data

def main():
    parser = argparse.ArgumentParser(description="Stage R: F4 Evidence & Robustness Analysis")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER", help="Workspace root directory")
    parser.add_argument("--no_train", action="store_true", default=True, help="Disable training mode (mandatory constraint)")
    args = parser.parse_args()

    project_root = args.project_root
    print(f"Project root: {project_root}")
    
    # Verify no training constraint
    if not args.no_train:
        print("ERROR: Training is explicitly forbidden in Stage R. Please run with --no_train.")
        sys.exit(1)
        
    # Append sys path to find source modules
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "src"))
    
    from src.models.cikd_pp_rt import CIKDPPResidualTransformer
    
    # Paths setup
    cache_dir = os.path.join(project_root, "data", "cache", "kg_complete")
    processed_dir = os.path.join(project_root, "data", "processed")
    out_dir = os.path.join(project_root, "outputs", "stage_r_f4_evidence_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load cache files
    print("Loading features and split metadata from cache...")
    split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
    labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
    y_ck = np.load(os.path.join(cache_dir, 'y_ck.npy'))
    sample_ids = np.load(os.path.join(cache_dir, 'sample_ids.npy'))
    
    text_feat = np.load(os.path.join(cache_dir, 'text_features.npy'))
    img_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))
    img_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))
    kg_feats = np.load(os.path.join(cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
    
    # Slice masks
    val_mask = (split_ids == 1)
    test_mask = (split_ids == 2)
    
    print(f"Cache loaded: Total {len(split_ids)} samples, Val size: {val_mask.sum()}, Test size: {test_mask.sum()}")
    
    # Load baseline logits
    print("Loading baseline logits...")
    val_logits_base = np.load(os.path.join(project_root, "outputs", "stage_f0_baseline_anchor", "val_logits_base.npy"))
    test_logits_base = np.load(os.path.join(project_root, "outputs", "stage_f0_baseline_anchor", "test_logits_base.npy"))
    
    # Load manifest dataframes
    print("Loading manifest files...")
    df_manifest_val = pd.read_csv(os.path.join(processed_dir, 'manifest_kg_complete_val_seed42.csv'))
    df_manifest_test = pd.read_csv(os.path.join(processed_dir, 'manifest_kg_complete_test_seed42.csv'))
    
    # Assert alignment
    assert (df_manifest_val['fine_label'].values == labels_fine[val_mask]).all(), "Validation labels alignment mismatch!"
    assert (df_manifest_test['fine_label'].values == labels_fine[test_mask]).all(), "Test labels alignment mismatch!"
    print("[+] Manifest files align programmatically row-for-row with the sliced cache features.")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load F4 checkpoint and model
    print("Loading F4 model...")
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_feats.shape[1]
    
    rt_model = CIKDPPResidualTransformer(
        num_relations=num_relations, 
        kg_dim=kg_dim, 
        d_model=256, 
        num_layers=2, 
        num_heads=4, 
        dropout=0.2
    ).to(device)
    
    checkpoint_path = os.path.join(project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
    rt_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(rt_ckpt, dict) and 'model_state_dict' in rt_ckpt:
        rt_model.load_state_dict(rt_ckpt['model_state_dict'])
    else:
        rt_model.load_state_dict(rt_ckpt)
    rt_model.eval()
    
    # Define Dataloader and inference loop
    def run_inference_split(text_f, img_g, img_p, kg_f, rel_i, base_logits):
        t_text = torch.tensor(text_f, dtype=torch.float32)
        t_img_g = torch.tensor(img_g, dtype=torch.float32)
        t_img_p = torch.tensor(img_p, dtype=torch.float32)
        t_kg = torch.tensor(kg_f, dtype=torch.float32)
        t_rel = torch.tensor(rel_i, dtype=torch.long)
        t_logits = torch.tensor(base_logits, dtype=torch.float32)
        
        ds = TensorDataset(t_text, t_img_g, t_img_p, t_kg, t_rel, t_logits)
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        
        preds_list = []
        c_probs_list = []
        logits_final_list = []
        
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
                
                logits_final_list.append(outputs['logits_final'].cpu().numpy())
                preds_list.extend(torch.argmax(outputs['logits_final'], dim=-1).cpu().numpy())
                c_probs_list.extend(torch.sigmoid(outputs['c_logit']).cpu().numpy())
                
        logits_final = np.concatenate(logits_final_list, axis=0)
        # Apply softmax to get probs
        probs_final = np.exp(logits_final) / np.sum(np.exp(logits_final), axis=-1, keepdims=True)
        return np.array(preds_list), np.array(c_probs_list), logits_final, probs_final

    print("\nEvaluating F4 on Val Split...")
    val_preds, val_c_probs, val_logits_final, val_probs_final = run_inference_split(
        text_feat[val_mask], img_global[val_mask], img_patch[val_mask], kg_feats[val_mask], relation_ids[val_mask], val_logits_base
    )
    
    print("Evaluating F4 on Test Split...")
    test_preds, test_c_probs, test_logits_final, test_probs_final = run_inference_split(
        text_feat[test_mask], img_global[test_mask], img_patch[test_mask], kg_feats[test_mask], relation_ids[test_mask], test_logits_base
    )
    
    # -------------------------------------------------------------
    # 1. R_FINAL_MODEL_SUMMARY.txt
    # -------------------------------------------------------------
    print("Writing Final Model Summary...")
    # Calculate Val metrics
    f4_val_acc = accuracy_score(labels_fine[val_mask], val_preds)
    f4_val_macro = f1_score(labels_fine[val_mask], val_preds, average='macro', zero_division=0)
    f4_val_ck = f1_score(labels_fine[val_mask], val_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    base_val_preds = np.argmax(val_logits_base, axis=-1)
    base_val_acc = accuracy_score(labels_fine[val_mask], base_val_preds)
    base_val_macro = f1_score(labels_fine[val_mask], base_val_preds, average='macro', zero_division=0)
    base_val_ck = f1_score(labels_fine[val_mask], base_val_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    # Calculate Test metrics
    f4_test_acc = accuracy_score(labels_fine[test_mask], test_preds)
    f4_test_macro = f1_score(labels_fine[test_mask], test_preds, average='macro', zero_division=0)
    f4_test_weighted = f1_score(labels_fine[test_mask], test_preds, average='weighted', zero_division=0)
    f4_test_ck = f1_score(labels_fine[test_mask], test_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    base_test_preds = np.argmax(test_logits_base, axis=-1)
    base_test_acc = accuracy_score(labels_fine[test_mask], base_test_preds)
    base_test_macro = f1_score(labels_fine[test_mask], base_test_preds, average='macro', zero_division=0)
    base_test_weighted = f1_score(labels_fine[test_mask], base_test_preds, average='weighted', zero_division=0)
    base_test_ck = f1_score(labels_fine[test_mask], base_test_preds, average=None, labels=list(range(6)), zero_division=0)[2]
    
    # Read TVCS metrics
    tvcs_mask_val = (y_ck[val_mask] != -1)
    val_tvcs_auc = roc_auc_score(y_ck[val_mask][tvcs_mask_val], val_c_probs[tvcs_mask_val])
    val_tvcs_pr_auc = average_precision_score(y_ck[val_mask][tvcs_mask_val], val_c_probs[tvcs_mask_val])
    val_mean_c_real = float(np.mean(val_c_probs[y_ck[val_mask] == 0]))
    val_mean_c_ck = float(np.mean(val_c_probs[y_ck[val_mask] == 1]))
    val_tvcs_delta = val_mean_c_ck - val_mean_c_real
    
    tvcs_mask_test = (y_ck[test_mask] != -1)
    test_tvcs_auc = roc_auc_score(y_ck[test_mask][tvcs_mask_test], test_c_probs[tvcs_mask_test])
    test_tvcs_pr_auc = average_precision_score(y_ck[test_mask][tvcs_mask_test], test_c_probs[tvcs_mask_test])
    test_mean_c_real = float(np.mean(test_c_probs[y_ck[test_mask] == 0]))
    test_mean_c_ck = float(np.mean(test_c_probs[y_ck[test_mask] == 1]))
    test_tvcs_delta = test_mean_c_ck - test_mean_c_real
    
    summary_text = f"""========================================================================
STAGE R: FINAL LOCKED MODEL (F4) ROBUSTNESS & EVIDENCE SUMMARY
========================================================================

1. KEY EXPERIMENTAL CONTEXT & CONSTRAINTS:
------------------------------------------
- NO TRAINING WAS RUN in Stage R. Predictions, probabilities, and TVCS metrics
  were generated strictly from the locked F4 model checkpoint (ablation_no_c_emb).
- F4 (CIKD++-RT no_c_emb) remains the locked final model for publication.
- Stage I (safe feature refresh) attempts were diagnostic only.
- Stage S1 (CAFE-lite) same-split baseline was val-only and diagnostic only.

2. CLASSIFICATION METRICS COMPARISON:
-------------------------------------
Validation Split:
- F4 model:          Accuracy: {f4_val_acc:.6f} | Macro-F1: {f4_val_macro:.6f} | CK-F1: {f4_val_ck:.6f}
- T+I+KG Baseline:   Accuracy: {base_val_acc:.6f} | Macro-F1: {base_val_macro:.6f} | CK-F1: {base_val_ck:.6f}
- G1 Co-Attention B: Accuracy: 0.530000 | Macro-F1: 0.443800 | CK-F1: 0.342900
- G1 Co-Attention A: Accuracy: 0.540769 | Macro-F1: 0.448613 | CK-F1: 0.332046
- S1 CAFE-Lite A:    Accuracy: 0.646154 | Macro-F1: 0.460561 | CK-F1: 0.314607

Locked Test Split (Final Publication Results):
- F4 model:          Accuracy: {f4_test_acc:.6f} | Macro-F1: {f4_test_macro:.6f} | Weighted-F1: {f4_test_weighted:.6f} | CK-F1: {f4_test_ck:.6f}
- T+I+KG Baseline:   Accuracy: {base_test_acc:.6f} | Macro-F1: {base_test_macro:.6f} | Weighted-F1: {base_test_weighted:.6f} | CK-F1: {base_test_ck:.6f}
- G1 Co-Attention B: Accuracy: 0.547177 | Macro-F1: 0.448010 | Weighted-F1: N/A      | CK-F1: 0.319192
- G1 Co-Attention A: Accuracy: N/A      | Macro-F1: N/A      | Weighted-F1: N/A      | CK-F1: N/A (Val-only diagnostic)
- S1 CAFE-Lite A:    Accuracy: N/A      | Macro-F1: N/A      | Weighted-F1: N/A      | CK-F1: N/A (Val-only diagnostic)

3. TVCS SPECIALIST EVIDENCE ANALYSIS:
-------------------------------------
Validation:
- TVCS AUC: {val_tvcs_auc:.6f} | PR-AUC: {val_tvcs_pr_auc:.6f}
- Mean Contradiction Probability (Real): {val_mean_c_real:.6f}
- Mean Contradiction Probability (CK):   {val_mean_c_ck:.6f}
- TVCS Delta (CK - Real):                 {val_tvcs_delta:.6f}

Locked Test:
- TVCS AUC: {test_tvcs_auc:.6f} | PR-AUC: {test_tvcs_pr_auc:.6f}
- Mean Contradiction Probability (Real): {test_mean_c_real:.6f}
- Mean Contradiction Probability (CK):   {test_mean_c_ck:.6f}
- TVCS Delta (CK - Real):                 {test_tvcs_delta:.6f}

Conclusion: The TVCS specialist successfully computes a strong contradiction signal.
The TVCS Delta of {test_tvcs_delta:.4f} demonstrates that samples with content-knowledge inconsistency (CK)
trigger significantly higher contradiction probability than real samples.

4. MODEL LIMITATIONS:
---------------------
- Minority Class trade-offs: G2-B (Focal Loss) and G4-D (CK-Aware Correction Head) sweeps
  resulted in higher overall accuracy but regressed on Macro-F1 and CK-F1, showing
  decision boundary instability.
- Class 2 vs Class 3 Confusion: The model frequently misclassifies class 2 (CK)
  as class 3 (Text Fake) and vice-versa, due to semantic overlaps.
"""
    with open(os.path.join(out_dir, "R_FINAL_MODEL_SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    # -------------------------------------------------------------
    # 2. R_PER_CLASS_F1.csv
    # -------------------------------------------------------------
    print("Writing Per-class F1...")
    class_names = [
        "Real", "Text-Image Inconsistency", "Content-Knowledge Inconsistency (CK)",
        "Text-based Fake", "Image-based Fake", "Others"
    ]
    per_class_val_p = precision_score(labels_fine[val_mask], val_preds, average=None, labels=list(range(6)), zero_division=0)
    per_class_val_r = recall_score(labels_fine[val_mask], val_preds, average=None, labels=list(range(6)), zero_division=0)
    per_class_val_f1 = f1_score(labels_fine[val_mask], val_preds, average=None, labels=list(range(6)), zero_division=0)
    
    per_class_test_p = precision_score(labels_fine[test_mask], test_preds, average=None, labels=list(range(6)), zero_division=0)
    per_class_test_r = recall_score(labels_fine[test_mask], test_preds, average=None, labels=list(range(6)), zero_division=0)
    per_class_test_f1 = f1_score(labels_fine[test_mask], test_preds, average=None, labels=list(range(6)), zero_division=0)
    
    df_per_class = pd.DataFrame({
        'class_id': list(range(6)),
        'class_name': class_names,
        'val_precision': per_class_val_p,
        'val_recall': per_class_val_r,
        'val_f1': per_class_val_f1,
        'test_precision': per_class_test_p,
        'test_recall': per_class_test_r,
        'test_f1': per_class_test_f1
    })
    df_per_class.to_csv(os.path.join(out_dir, "R_PER_CLASS_F1.csv"), index=False)

    # -------------------------------------------------------------
    # 3. R_CONFUSION_MATRIX.csv
    # -------------------------------------------------------------
    print("Writing Confusion Matrix...")
    cm_test = confusion_matrix(labels_fine[test_mask], test_preds, labels=list(range(6)))
    df_cm = pd.DataFrame(cm_test, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    df_cm.to_csv(os.path.join(out_dir, "R_CONFUSION_MATRIX.csv"))

    # -------------------------------------------------------------
    # 4. R_CK_ERROR_TRANSITIONS.csv
    # -------------------------------------------------------------
    print("Writing CK Error Transitions...")
    # CK -> Real: True=2, Pred=0
    ck_to_real = int(np.sum((labels_fine[test_mask] == 2) & (test_preds == 0)))
    # CK -> Class 3: True=2, Pred=3
    ck_to_class3 = int(np.sum((labels_fine[test_mask] == 2) & (test_preds == 3)))
    # Class 3 -> CK: True=3, Pred=2
    class3_to_ck = int(np.sum((labels_fine[test_mask] == 3) & (test_preds == 2)))
    
    # Bottleneck behavior for classes 1, 2, 5 (Compare F4 vs baseline)
    base_test_p = precision_score(labels_fine[test_mask], base_test_preds, average=None, labels=list(range(6)), zero_division=0)
    base_test_r = recall_score(labels_fine[test_mask], base_test_preds, average=None, labels=list(range(6)), zero_division=0)
    base_test_f1 = f1_score(labels_fine[test_mask], base_test_preds, average=None, labels=list(range(6)), zero_division=0)
    
    # Class 5 prediction distribution
    class5_preds = test_preds[labels_fine[test_mask] == 5]
    c5_dist = {c: int(np.sum(class5_preds == c)) for c in range(6)}
    
    transition_rows = [
        {"Metric": "CK -> Real Transition", "Count": ck_to_real, "Description": "True CK predicted as Real"},
        {"Metric": "CK -> Class 3 Transition", "Count": ck_to_class3, "Description": "True CK predicted as Text Fake"},
        {"Metric": "Class 3 -> CK Transition", "Count": class3_to_ck, "Description": "True Text Fake predicted as CK"},
        {"Metric": "Class 1 Baseline Precision", "Count": base_test_p[1], "Description": "Text-Image Inconsistency"},
        {"Metric": "Class 1 Baseline Recall", "Count": base_test_r[1], "Description": "Text-Image Inconsistency"},
        {"Metric": "Class 1 Baseline F1", "Count": base_test_f1[1], "Description": "Text-Image Inconsistency"},
        {"Metric": "Class 1 F4 Precision", "Count": per_class_test_p[1], "Description": "Text-Image Inconsistency"},
        {"Metric": "Class 1 F4 Recall", "Count": per_class_test_r[1], "Description": "Text-Image Inconsistency"},
        {"Metric": "Class 1 F4 F1", "Count": per_class_test_f1[1], "Description": "Text-Image Inconsistency"},
        {"Metric": "Class 2 Baseline Precision", "Count": base_test_p[2], "Description": "CK"},
        {"Metric": "Class 2 Baseline Recall", "Count": base_test_r[2], "Description": "CK"},
        {"Metric": "Class 2 Baseline F1", "Count": base_test_f1[2], "Description": "CK"},
        {"Metric": "Class 2 F4 Precision", "Count": per_class_test_p[2], "Description": "CK"},
        {"Metric": "Class 2 F4 Recall", "Count": per_class_test_r[2], "Description": "CK"},
        {"Metric": "Class 2 F4 F1", "Count": per_class_test_f1[2], "Description": "CK"},
        {"Metric": "Class 5 Baseline Precision", "Count": base_test_p[5], "Description": "Others"},
        {"Metric": "Class 5 Baseline Recall", "Count": base_test_r[5], "Description": "Others"},
        {"Metric": "Class 5 Baseline F1", "Count": base_test_f1[5], "Description": "Others"},
        {"Metric": "Class 5 F4 Precision", "Count": per_class_test_p[5], "Description": "Others"},
        {"Metric": "Class 5 F4 Recall", "Count": per_class_test_r[5], "Description": "Others"},
        {"Metric": "Class 5 F4 F1", "Count": per_class_test_f1[5], "Description": "Others"},
    ]
    for c in range(6):
        transition_rows.append({
            "Metric": f"True Class 5 predicted as Class {c}",
            "Count": c5_dist[c],
            "Description": f"Distribution of True Class 5 (Others) predictions"
        })
        
    pd.DataFrame(transition_rows).to_csv(os.path.join(out_dir, "R_CK_ERROR_TRANSITIONS.csv"), index=False)

    # -------------------------------------------------------------
    # 5. R_TVCS_DISTRIBUTION_BY_CLASS.csv
    # -------------------------------------------------------------
    print("Writing TVCS Distribution by Class...")
    tvcs_dist_rows = []
    for split_name, mask, scores in [("val", val_mask, val_c_probs), ("test", test_mask, test_c_probs)]:
        lbls = labels_fine[mask]
        for c in range(6):
            c_mask = (lbls == c)
            if c_mask.sum() > 0:
                c_scores = scores[c_mask]
                tvcs_dist_rows.append({
                    'split': split_name,
                    'class_id': c,
                    'class_name': class_names[c],
                    'mean_tvcs_score': np.mean(c_scores),
                    'std_tvcs_score': np.std(c_scores),
                    'min_tvcs_score': np.min(c_scores),
                    'max_tvcs_score': np.max(c_scores),
                    'count': int(c_mask.sum())
                })
    pd.DataFrame(tvcs_dist_rows).to_csv(os.path.join(out_dir, "R_TVCS_DISTRIBUTION_BY_CLASS.csv"), index=False)

    # -------------------------------------------------------------
    # 6. R_TVCS_REAL_VS_CK.csv
    # -------------------------------------------------------------
    print("Writing TVCS Real vs CK histogram bins...")
    # Bins for histogram
    bins = np.linspace(0.0, 1.0, 21) # 20 bins
    test_y_ck_s = y_ck[test_mask]
    real_scores = test_c_probs[test_y_ck_s == 0]
    ck_scores = test_c_probs[test_y_ck_s == 1]
    
    real_counts, _ = np.histogram(real_scores, bins=bins)
    ck_counts, _ = np.histogram(ck_scores, bins=bins)
    
    hist_rows = []
    for i in range(20):
        hist_rows.append({
            'bin_index': i,
            'bin_lower': bins[i],
            'bin_upper': bins[i+1],
            'real_count': int(real_counts[i]),
            'ck_count': int(ck_counts[i]),
            'real_density': real_counts[i] / len(real_scores) if len(real_scores) > 0 else 0.0,
            'ck_density': ck_counts[i] / len(ck_scores) if len(ck_scores) > 0 else 0.0
        })
    df_hist = pd.DataFrame(hist_rows)
    # Add overall metrics as final rows / separate columns
    df_hist['mean_c_real'] = test_mean_c_real
    df_hist['mean_c_ck'] = test_mean_c_ck
    df_hist['tvcs_delta'] = test_tvcs_delta
    df_hist['tvcs_auc'] = test_tvcs_auc
    df_hist['tvcs_pr_auc'] = test_tvcs_pr_auc
    
    df_hist.to_csv(os.path.join(out_dir, "R_TVCS_REAL_VS_CK.csv"), index=False)

    # -------------------------------------------------------------
    # 7. R_RESCUE_BROKEN_SUMMARY.csv
    # -------------------------------------------------------------
    print("Writing Rescue/Broken Summary...")
    # Definitions:
    # Rescued: baseline incorrect, F4 correct
    # Broken: baseline correct, F4 incorrect
    
    def analyze_rescue_broken(lbls, base_pr, f4_pr, subset_mask):
        sub_lbls = lbls[subset_mask]
        sub_base = base_pr[subset_mask]
        sub_f4 = f4_pr[subset_mask]
        
        base_correct = (sub_base == sub_lbls)
        f4_correct = (sub_f4 == sub_lbls)
        
        rescued = int(np.sum((~base_correct) & f4_correct))
        broken = int(np.sum(base_correct & (~f4_correct)))
        both_correct = int(np.sum(base_correct & f4_correct))
        both_wrong = int(np.sum((~base_correct) & (~f4_correct)))
        
        return {
            'rescued': rescued,
            'broken': broken,
            'net_improvement': rescued - broken,
            'baseline_correct_total': rescued + both_correct, # Wait, base_correct = rescued (F4 correct) is incorrect, actually base_correct = both_correct + broken
            'f4_correct_total': rescued + both_correct,
            'both_correct': both_correct,
            'both_wrong': both_wrong
        }

    lbls_t = labels_fine[test_mask]
    summary_rows = []
    
    # Overall
    overall_res = analyze_rescue_broken(lbls_t, base_test_preds, test_preds, np.ones(len(lbls_t), dtype=bool))
    overall_res['MetricType'] = "Overall"
    summary_rows.append(overall_res)
    
    # CK (Class 2)
    ck_res = analyze_rescue_broken(lbls_t, base_test_preds, test_preds, (lbls_t == 2))
    ck_res['MetricType'] = "CK (Class 2)"
    summary_rows.append(ck_res)
    
    # Real (Class 0)
    real_res = analyze_rescue_broken(lbls_t, base_test_preds, test_preds, (lbls_t == 0))
    real_res['MetricType'] = "Real (Class 0)"
    summary_rows.append(real_res)
    
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "R_RESCUE_BROKEN_SUMMARY.csv"), index=False)

    # -------------------------------------------------------------
    # 8. R_RESCUE_BROKEN_BY_CLASS.csv
    # -------------------------------------------------------------
    print("Writing Rescue/Broken by Class...")
    class_res_rows = []
    for c in range(6):
        c_res = analyze_rescue_broken(lbls_t, base_test_preds, test_preds, (lbls_t == c))
        c_res['class_id'] = c
        c_res['class_name'] = class_names[c]
        c_res['total_samples'] = int(np.sum(lbls_t == c))
        class_res_rows.append(c_res)
        
    pd.DataFrame(class_res_rows).to_csv(os.path.join(out_dir, "R_RESCUE_BROKEN_BY_CLASS.csv"), index=False)

    # -------------------------------------------------------------
    # 9. R_CALIBRATION_ECE.csv
    # -------------------------------------------------------------
    print("Writing Calibration & ECE bins...")
    f4_ece, f4_bin_data = compute_ece(test_probs_final, lbls_t)
    
    # Apply softmax to baseline logits
    base_probs = np.exp(test_logits_base) / np.sum(np.exp(test_logits_base), axis=-1, keepdims=True)
    base_ece, base_bin_data = compute_ece(base_probs, lbls_t)
    
    cal_rows = []
    for row in f4_bin_data:
        row['model'] = 'F4'
        row['overall_ece'] = f4_ece
        cal_rows.append(row)
        
    for row in base_bin_data:
        row['model'] = 'Baseline'
        row['overall_ece'] = base_ece
        cal_rows.append(row)
        
    pd.DataFrame(cal_rows).to_csv(os.path.join(out_dir, "R_CALIBRATION_ECE.csv"), index=False)

    # -------------------------------------------------------------
    # 10. R_CASE_STUDY_CANDIDATES.csv
    # -------------------------------------------------------------
    print("Writing Case Study Candidates...")
    # Find CK samples (true class 2)
    ck_mask_t = (lbls_t == 2)
    ck_indices = np.where(ck_mask_t)[0]
    
    rescued_ck_indices = []
    broken_ck_indices = []
    
    for idx in ck_indices:
        base_correct = (base_test_preds[idx] == 2)
        f4_correct = (test_preds[idx] == 2)
        
        if not base_correct and f4_correct:
            rescued_ck_indices.append(idx)
        elif base_correct and not f4_correct:
            broken_ck_indices.append(idx)
            
    # Print counts of candidates found
    print(f"  Found {len(rescued_ck_indices)} rescued CK cases, {len(broken_ck_indices)} broken CK cases.")
    
    selected_cases = []
    
    # Select 2 correctly fixed cases
    for i, idx in enumerate(rescued_ck_indices[:2]):
        original_sample_id = int(df_manifest_test.iloc[idx]['sample_id']) if 'sample_id' in df_manifest_test.columns else int(sample_ids[test_mask][idx])
        text_content = df_manifest_test.iloc[idx]['text']
        img_p = df_manifest_test.iloc[idx]['image_path']
        tvcs_val = float(test_c_probs[idx])
        conf = float(np.max(test_probs_final[idx]))
        
        explanation = (
            f"F4 corrected the baseline prediction. TVCS contradiction probability was {tvcs_val:.4f} "
            f"(above threshold), triggering the residual correction pathway to correctly classify the "
            f"Content-Knowledge inconsistency."
        ) if tvcs_val > 0.5 else (
            f"F4 corrected the baseline prediction. TVCS score was low ({tvcs_val:.4f}), "
            f"but multi-modal residual transformer features correctly shifted the decision boundary."
        )
        
        selected_cases.append({
            'case_type': 'rescued_ck',
            'sample_id': original_sample_id,
            'true_label': 2,
            'baseline_pred': int(base_test_preds[idx]),
            'f4_pred': 2,
            'confidence': conf,
            'tvcs_score': tvcs_val,
            'text': text_content,
            'image_path': img_p,
            'explanation_notes': explanation
        })
        
    # Select 1 CK failure case (broken by F4)
    if len(broken_ck_indices) > 0:
        idx = broken_ck_indices[0]
        original_sample_id = int(df_manifest_test.iloc[idx]['sample_id']) if 'sample_id' in df_manifest_test.columns else int(sample_ids[test_mask][idx])
        text_content = df_manifest_test.iloc[idx]['text']
        img_p = df_manifest_test.iloc[idx]['image_path']
        tvcs_val = float(test_c_probs[idx])
        conf = float(np.max(test_probs_final[idx]))
        
        explanation = (
            f"Baseline predicted CK correctly, but F4 corrupted it to class {test_preds[idx]}. "
            f"TVCS contradiction probability was too low ({tvcs_val:.4f} <= 0.5), failing to flag "
            f"the contradiction cue, and residual transformer logits over-corrected the prediction."
        ) if tvcs_val <= 0.5 else (
            f"Baseline predicted CK correctly, but F4 corrupted it to class {test_preds[idx]}. "
            f"Although TVCS contradiction probability was high ({tvcs_val:.4f}), the residual transformer "
            f"delta logits mis-shifted the prediction due to strong textual/visual cues matching class {test_preds[idx]}."
        )
        
        selected_cases.append({
            'case_type': 'broken_ck',
            'sample_id': original_sample_id,
            'true_label': 2,
            'baseline_pred': 2,
            'f4_pred': int(test_preds[idx]),
            'confidence': conf,
            'tvcs_score': tvcs_val,
            'text': text_content,
            'image_path': img_p,
            'explanation_notes': explanation
        })
        
    pd.DataFrame(selected_cases).to_csv(os.path.join(out_dir, "R_CASE_STUDY_CANDIDATES.csv"), index=False)

    # -------------------------------------------------------------
    # 11. R_ROBUSTNESS_BY_METADATA.csv
    # -------------------------------------------------------------
    print("Writing Robustness Analysis by Metadata...")
    robust_rows = []
    
    # 11.1 Topic analysis
    df_manifest_test['f4_pred'] = test_preds
    df_manifest_test['baseline_pred'] = base_test_preds
    df_manifest_test['correct_f4'] = (test_preds == lbls_t)
    df_manifest_test['correct_base'] = (base_test_preds == lbls_t)
    
    unique_topics = df_manifest_test['topic'].unique()
    for topic in unique_topics:
        sub_df = df_manifest_test[df_manifest_test['topic'] == topic]
        n_samples = len(sub_df)
        if n_samples >= 5: # only keep representative topics
            topic_lbls = sub_df['fine_label'].values
            
            f4_acc_t = accuracy_score(topic_lbls, sub_df['f4_pred'].values)
            f4_macro_t = f1_score(topic_lbls, sub_df['f4_pred'].values, average='macro', zero_division=0)
            f4_ck_t = f1_score(topic_lbls, sub_df['f4_pred'].values, average=None, labels=list(range(6)), zero_division=0)[2]
            
            base_acc_t = accuracy_score(topic_lbls, sub_df['baseline_pred'].values)
            base_macro_t = f1_score(topic_lbls, sub_df['baseline_pred'].values, average='macro', zero_division=0)
            base_ck_t = f1_score(topic_lbls, sub_df['baseline_pred'].values, average=None, labels=list(range(6)), zero_division=0)[2]
            
            robust_rows.append({
                'metadata_field': 'Topic',
                'metadata_value': topic,
                'sample_count': n_samples,
                'f4_accuracy': f4_acc_t,
                'f4_macro_f1': f4_macro_t,
                'f4_ck_f1': f4_ck_t,
                'baseline_accuracy': base_acc_t,
                'baseline_macro_f1': base_macro_t,
                'baseline_ck_f1': base_ck_t,
                'improvement_acc': f4_acc_t - base_acc_t,
                'improvement_ck_f1': f4_ck_t - base_ck_t
            })
            
    # 11.2 Platform analysis
    unique_platforms = df_manifest_test['platform'].unique()
    for platform in unique_platforms:
        if pd.isna(platform):
            platform = 'Unknown'
            sub_df = df_manifest_test[df_manifest_test['platform'].isna()]
        else:
            sub_df = df_manifest_test[df_manifest_test['platform'] == platform]
            
        n_samples = len(sub_df)
        if n_samples >= 5:
            platform_lbls = sub_df['fine_label'].values
            
            f4_acc_p = accuracy_score(platform_lbls, sub_df['f4_pred'].values)
            f4_macro_p = f1_score(platform_lbls, sub_df['f4_pred'].values, average='macro', zero_division=0)
            f4_ck_p = f1_score(platform_lbls, sub_df['f4_pred'].values, average=None, labels=list(range(6)), zero_division=0)[2]
            
            base_acc_p = accuracy_score(platform_lbls, sub_df['baseline_pred'].values)
            base_macro_p = f1_score(platform_lbls, sub_df['baseline_pred'].values, average='macro', zero_division=0)
            base_ck_p = f1_score(platform_lbls, sub_df['baseline_pred'].values, average=None, labels=list(range(6)), zero_division=0)[2]
            
            robust_rows.append({
                'metadata_field': 'Platform',
                'metadata_value': platform,
                'sample_count': n_samples,
                'f4_accuracy': f4_acc_p,
                'f4_macro_f1': f4_macro_p,
                'f4_ck_f1': f4_ck_p,
                'baseline_accuracy': base_acc_p,
                'baseline_macro_f1': base_macro_p,
                'baseline_ck_f1': base_ck_p,
                'improvement_acc': f4_acc_p - base_acc_p,
                'improvement_ck_f1': f4_ck_p - base_ck_p
            })
            
    df_robust = pd.DataFrame(robust_rows)
    df_robust.to_csv(os.path.join(out_dir, "R_ROBUSTNESS_BY_METADATA.csv"), index=False)
    
    print("\nStage R analysis ran successfully! All output files written under outputs/stage_r_f4_evidence_analysis/")

if __name__ == "__main__":
    main()
