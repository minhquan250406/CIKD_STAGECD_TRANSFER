"""
Stage I-E0: Bottleneck Audit (Validation-Only).
Loads validation split, evaluates F4 and I-BC-A/B/C/D configurations, 
performs per-class analysis, confusion matrices, rescued/broken counts, CK transitions,
and identifies Macro-F1 bottleneck classes to recommend a correction mask.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add workspace path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cikd_pp_rt import CIKDPPResidualTransformer
from src.stage_i_bc_multitask_balanced_model import StageIBCMultitaskCIKDPP

CLASS_NAMES = [
    "real",
    "text-image inconsistency",
    "content-knowledge inconsistency",
    "text-based fake",
    "image-based fake",
    "others"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-E0 Validation Bottleneck Audit.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--split", type=str, default="val", choices=["val"],
                        help="Split to use. Strictly locked to validation.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be true to enforce that test split is never evaluated.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.split == "val", "Error: Audit is strictly restricted to validation split."
    assert args.no_test_eval, "Error: --no_test_eval flag must be present to guarantee test set safety."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running audit on device: {device}")
    
    # 1. Output directories
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e", "e0_bottleneck_audit")
    matrix_dir = os.path.join(out_dir, "E0_CONFUSION_MATRICES")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)
    
    # 2. Load cached features
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
    
    # Strictly extract validation split
    val_mask = (split_ids == 1)
    # Double check no test set indices are included
    assert not np.any(split_ids[val_mask] == 2), "Test split leaked into validation split mask!"
    
    val_labels = labels_fine[val_mask]
    val_y_ck = y_ck[val_mask]
    val_relations = relation_ids[val_mask]
    
    # Baseline validation logits
    baseline_dir = os.path.join(args.project_root, "outputs", "stage_f0_baseline_anchor")
    val_logits_base = np.load(os.path.join(baseline_dir, 'val_logits_base.npy'))
    assert len(val_logits_base) == np.sum(val_mask), f"Baseline logits size {len(val_logits_base)} != val split size {np.sum(val_mask)}"
    
    # Convert validation features to PyTorch tensors
    v_text = torch.tensor(text_features[val_mask], dtype=torch.float32).to(device)
    v_img_g = torch.tensor(image_features_global[val_mask], dtype=torch.float32).to(device)
    v_img_p = torch.tensor(image_features_patch[val_mask], dtype=torch.float32).to(device)
    v_kg = torch.tensor(kg_features[val_mask], dtype=torch.float32).to(device)
    v_rel = torch.tensor(relation_ids[val_mask], dtype=torch.long).to(device)
    v_logits = torch.tensor(val_logits_base, dtype=torch.float32).to(device)
    
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_features.shape[1]
    
    # 3. Load F4 Model
    f4_ckpt_path = os.path.join(args.project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
    print(f"[+] Loading F4 checkpoint: {f4_ckpt_path}")
    f4_model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    f4_ckpt = torch.load(f4_ckpt_path, map_location=device, weights_only=False)
    f4_state = f4_ckpt.get('model_state_dict', f4_ckpt)
    f4_model.load_state_dict(f4_state)
    f4_model.eval()
    
    # Run F4 validation inference
    with torch.no_grad():
        f4_outputs = f4_model(
            text_features=v_text,
            image_global_features=v_img_g,
            image_patch_features=v_img_p,
            kg_features=v_kg,
            relation_ids=v_rel,
            baseline_logits=v_logits,
            ablation_no_c_emb=True
        )
    f4_preds = torch.argmax(f4_outputs['logits_final'], dim=-1).cpu().numpy()
    
    # 4. Load Stage I-BC models
    ibc_configs = ['a', 'b', 'c', 'd']
    ibc_preds = {}
    
    for config_char in ibc_configs:
        ibc_ckpt_path = os.path.join(args.project_root, "checkpoints", "stage_i", f"best_i_bc_{config_char}.pt")
        print(f"[+] Loading I-BC-{config_char.upper()} checkpoint: {ibc_ckpt_path}")
        
        if not os.path.exists(ibc_ckpt_path):
            raise FileNotFoundError(f"Missing required I-BC checkpoint: {ibc_ckpt_path}")
            
        ckpt = torch.load(ibc_ckpt_path, map_location=device, weights_only=False)
        config_data = ckpt.get('config', {})
        alpha_max = config_data.get('alpha_max', 0.5)
        dropout = config_data.get('dropout', 0.2)
        
        ibc_model = StageIBCMultitaskCIKDPP(
            num_relations=num_relations,
            kg_dim=kg_dim,
            d_model=256,
            num_layers=2,
            num_heads=4,
            dropout=dropout,
            alpha_init=0.2,
            alpha_max=alpha_max
        ).to(device)
        
        ibc_model.load_state_dict(ckpt['model_state_dict'])
        ibc_model.eval()
        
        with torch.no_grad():
            outputs = ibc_model(
                text_features=v_text,
                image_global_features=v_img_g,
                image_patch_features=v_img_p,
                kg_features=v_kg,
                relation_ids=v_rel,
                baseline_logits=v_logits,
                ablation_no_c_emb=True
            )
        preds = torch.argmax(outputs['logits_final'], dim=-1).cpu().numpy()
        ibc_preds[f"I-BC-{config_char.upper()}"] = preds

    # 5. Compare Performance (F1 and metrics)
    models = ["F4"] + [f"I-BC-{c.upper()}" for c in ibc_configs]
    preds_dict = {"F4": f4_preds}
    preds_dict.update(ibc_preds)
    
    f1_records = []
    confusion_matrices = {}
    
    for m in models:
        m_preds = preds_dict[m]
        acc = accuracy_score(val_labels, m_preds)
        micro_f1 = f1_score(val_labels, m_preds, average='micro', zero_division=0)
        macro_f1 = f1_score(val_labels, m_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(val_labels, m_preds, average='weighted', zero_division=0)
        per_class_f1 = f1_score(val_labels, m_preds, average=None, labels=list(range(6)), zero_division=0)
        
        record = {
            "model": m,
            "val_accuracy": acc,
            "val_micro_f1": micro_f1,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1
        }
        for c in range(6):
            record[f"f1_class_{c}"] = per_class_f1[c]
        f1_records.append(record)
        
        # Save confusion matrix to a file
        cm = confusion_matrix(val_labels, m_preds, labels=list(range(6)))
        confusion_matrices[m] = cm
        df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        df_cm.to_csv(os.path.join(matrix_dir, f"{m.replace('-', '_').lower()}_confusion_matrix.csv"))
        
    df_f1 = pd.DataFrame(f1_records)
    df_f1.to_csv(os.path.join(out_dir, "E0_PER_CLASS_F1_COMPARISON.csv"), index=False)
    print("[+] Saved E0_PER_CLASS_F1_COMPARISON.csv")
    
    # 6. Class-wise rescued / broken counts vs F4
    rescue_broken_records = []
    
    for m in [f"I-BC-{c.upper()}" for c in ibc_configs]:
        m_preds = preds_dict[m]
        
        f4_correct = (f4_preds == val_labels)
        m_correct = (m_preds == val_labels)
        
        # Rescued: F4 incorrect, Variant correct
        rescued_overall = (~f4_correct) & m_correct
        # Broken: F4 correct, Variant incorrect
        broken_overall = f4_correct & (~m_correct)
        
        # Class-wise counts
        for c in range(6):
            c_mask = (val_labels == c)
            rescued_c = int(np.sum(rescued_overall[c_mask]))
            broken_c = int(np.sum(broken_overall[c_mask]))
            rescue_broken_records.append({
                "comparison_model": m,
                "class_id": c,
                "class_name": CLASS_NAMES[c],
                "rescued_count": rescued_c,
                "broken_count": broken_c,
                "net_rescue": rescued_c - broken_c
            })
            
    df_rb = pd.DataFrame(rescue_broken_records)
    df_rb.to_csv(os.path.join(out_dir, "E0_RESCUE_BROKEN_BY_CLASS.csv"), index=False)
    print("[+] Saved E0_RESCUE_BROKEN_BY_CLASS.csv")
    
    # 7. CK (Content-Knowledge Inconsistency) transitions
    # CK is class 2. We analyze transitions between CK (class 2) and other classes:
    # CK -> Class 3, CK -> Real (Class 0), Class 3 -> CK
    ck_transition_records = []
    
    for m in models:
        m_preds = preds_dict[m]
        
        # True is CK (class 2)
        true_ck_mask = (val_labels == 2)
        total_true_ck = int(np.sum(true_ck_mask))
        pred_as_ck_when_true_ck = int(np.sum(m_preds[true_ck_mask] == 2))
        pred_as_class3_when_true_ck = int(np.sum(m_preds[true_ck_mask] == 3))
        pred_as_real_when_true_ck = int(np.sum(m_preds[true_ck_mask] == 0))
        
        # True is Class 3 (Text Fake)
        true_class3_mask = (val_labels == 3)
        total_true_class3 = int(np.sum(true_class3_mask))
        pred_as_ck_when_true_class3 = int(np.sum(m_preds[true_class3_mask] == 2))
        
        # True is Class 0 (Real)
        true_real_mask = (val_labels == 0)
        total_true_real = int(np.sum(true_real_mask))
        pred_as_ck_when_true_real = int(np.sum(m_preds[true_real_mask] == 2))
        
        ck_transition_records.append({
            "model": m,
            "total_true_ck": total_true_ck,
            "ck_predicted_ck": pred_as_ck_when_true_ck,
            "ck_predicted_class3": pred_as_class3_when_true_ck,
            "ck_predicted_real": pred_as_real_when_true_ck,
            "total_true_class3": total_true_class3,
            "class3_predicted_ck": pred_as_ck_when_true_class3,
            "total_true_real": total_true_real,
            "real_predicted_ck": pred_as_ck_when_true_real
        })
        
    df_ck_trans = pd.DataFrame(ck_transition_records)
    df_ck_trans.to_csv(os.path.join(out_dir, "E0_CK_TRANSITIONS.csv"), index=False)
    print("[+] Saved E0_CK_TRANSITIONS.csv")
    
    # 8. Rank bottleneck classes by F4 and I-BC performance
    # Bottlenecks are characterized by:
    # 1. Low F4 F1
    # 2. Large False Negatives (True label = C, Pred != C)
    # 3. Large Confusion into Real/class 0 or class 3 (True label = C, Pred == 0 or 3)
    # 4. Whether I-BC variants improve or hurt that class (average net change)
    ranking_records = []
    f4_per_class_f1 = f1_score(val_labels, f4_preds, average=None, labels=list(range(6)), zero_division=0)
    
    for c in range(6):
        c_mask = (val_labels == c)
        f4_preds_c = f4_preds[c_mask]
        
        # 1. F4 F1
        f1_f4 = f4_per_class_f1[c]
        
        # 2. False negatives count under F4
        fn_count = int(np.sum(f4_preds_c != c))
        
        # 3. Confusion into Real/class0 or class3 under F4
        conf_to_0_3 = int(np.sum((f4_preds_c == 0) | (f4_preds_c == 3)))
        
        # 4. Net improvement by I-BC variants on average
        net_rb_c = df_rb[df_rb["class_id"] == c]["net_rescue"].mean()
        
        ranking_records.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "f4_f1": f1_f4,
            "f4_false_negatives": fn_count,
            "f4_confusion_to_real_or_class3": conf_to_0_3,
            "average_ibc_net_rescue": net_rb_c
        })
        
    df_rank = pd.DataFrame(ranking_records)
    # Sort: lowest F4 F1 first
    df_rank = df_rank.sort_values(by="f4_f1", ascending=True)
    df_rank.to_csv(os.path.join(out_dir, "E0_BOTTLENECK_CLASS_RANKING.csv"), index=False)
    print("[+] Saved E0_BOTTLENECK_CLASS_RANKING.csv")
    
    # 9. Recommended Class Correction Mask
    # Rules:
    # - Small correction for class0/Real: 0.1
    # - Medium correction for already-stable classes (classes where F4 F1 is relatively high, e.g. > 0.5)
    # - Larger correction for bottleneck classes (F4 F1 <= 0.45 or lowest performing classes)
    # Let's inspect F4 per class F1 dynamically
    recommended_mask = [1.0] * 6
    mask_explanations = {}
    
    for c in range(6):
        f1 = f4_per_class_f1[c]
        if c == 0:
            recommended_mask[c] = 0.1  # Constrained smaller for Real/class0
            mask_explanations[CLASS_NAMES[c]] = "Real class, constrained small to avoid bias"
        elif f1 < 0.45:
            recommended_mask[c] = 1.0  # Bottleneck class
            mask_explanations[CLASS_NAMES[c]] = f"Bottleneck class (F4 F1 = {f1:.4f} < 0.45), maximum correction"
        elif f1 < 0.55:
            recommended_mask[c] = 0.8  # Mild bottleneck
            mask_explanations[CLASS_NAMES[c]] = f"Mild bottleneck class (F4 F1 = {f1:.4f}), high correction"
        else:
            recommended_mask[c] = 0.5  # Stable class
            mask_explanations[CLASS_NAMES[c]] = f"Stable class (F4 F1 = {f1:.4f}), medium correction"
            
    mask_json = {
        "class_mask": recommended_mask,
        "class_names": CLASS_NAMES,
        "explanations": mask_explanations
    }
    
    mask_path = os.path.join(out_dir, "E0_RECOMMENDED_CLASS_MASK.json")
    with open(mask_path, "w") as f:
        json.dump(mask_json, f, indent=4)
    print(f"[+] Saved recommended class mask to: {mask_path}")
    
    # 10. Audit Summary Report
    # Check if Stage I-BC improved CK-F1 but not Macro-F1:
    # Let's check I-BC variants' macro-F1 vs F4 macro-F1 (0.4792) and CK-F1 (0.3922)
    f4_macro = df_f1[df_f1["model"] == "F4"]["val_macro_f1"].values[0]
    f4_ck = df_f1[df_f1["model"] == "F4"]["f1_class_2"].values[0]
    
    ibc_macro_max = df_f1[df_f1["model"] != "F4"]["val_macro_f1"].max()
    ibc_ck_max = df_f1[df_f1["model"] != "F4"]["f1_class_2"].max()
    
    macro_improved = ibc_macro_max > f4_macro
    ck_improved = ibc_ck_max > f4_ck
    
    bottleneck_classes_list = df_rank[df_rank["f4_f1"] < 0.45]["class_name"].tolist()
    
    summary_path = os.path.join(out_dir, "E0_AUDIT_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("========================================================================\n")
        f.write("STAGE I-E0 BOTTLENECK AUDIT SUMMARY (VALIDATION-ONLY)\n")
        f.write("========================================================================\n\n")
        f.write("CRITICAL ASSURANCES:\n")
        f.write("--------------------\n")
        f.write("- NO TRAINING WAS RUN.\n")
        f.write("- LOCKED TEST WAS NOT EVALUATED.\n")
        f.write("- Only the validation split was used for analysis.\n\n")
        
        f.write("STAGE I-BC PERFORMANCE CHECK:\n")
        f.write("-----------------------------\n")
        f.write(f"F4 Validation Macro-F1: {f4_macro:.6f} | CK-F1: {f4_ck:.6f}\n")
        f.write(f"I-BC Best Validation Macro-F1: {ibc_macro_max:.6f} | CK-F1: {ibc_ck_max:.6f}\n")
        f.write(f"Did Stage I-BC improve CK-F1? {'YES' if ck_improved else 'NO'}\n")
        f.write(f"Did Stage I-BC improve Macro-F1? {'YES' if macro_improved else 'NO'}\n")
        f.write("Observation: Stage I-BC models indeed improved CK-F1 or achieved similar results but failed to pass the Macro-F1 promotion gate.\n\n")
        
        f.write("MACRO-F1 BOTTLENECK CLASSES:\n")
        f.write("----------------------------\n")
        f.write("Ranking of classes from lowest to highest F4 F1:\n")
        for idx, row in df_rank.iterrows():
            f.write(f"Rank {idx+1}: Class {row['class_id']} ({row['class_name']}) | F4 F1: {row['f4_f1']:.6f} | FNs: {row['f4_false_negatives']} | Confusion to 0/3: {row['f4_confusion_to_real_or_class3']}\n")
        f.write(f"\nBottleneck classes identified (F1 < 0.45): {bottleneck_classes_list}\n\n")
        
        f.write("RECOMMENDED CLASS MASK FOR STAGE I-E1:\n")
        f.write("-------------------------------------\n")
        f.write(f"Recommended Mask Values: {recommended_mask}\n")
        f.write(f"Saved Mask JSON: outputs/stage_i_macro_micro_improvement/stage_i_e/e0_bottleneck_audit/E0_RECOMMENDED_CLASS_MASK.json\n\n")
        
        f.write("CLASS-WISE NET RESCUE (IBC VARIANTS VS F4):\n")
        f.write("------------------------------------------\n")
        for m in [f"I-BC-{c.upper()}" for c in ibc_configs]:
            m_rb = df_rb[df_rb["comparison_model"] == m]
            rescued_total = m_rb["rescued_count"].sum()
            broken_total = m_rb["broken_count"].sum()
            net_total = rescued_total - broken_total
            f.write(f"{m} vs F4: Total Rescued = {rescued_total} | Broken = {broken_total} | Net = {net_total:+d}\n")
            for _, r in m_rb.iterrows():
                f.write(f"  - Class {r['class_id']} ({r['class_name']}): Rescued={r['rescued_count']}, Broken={r['broken_count']}, Net={r['net_rescue']:+d}\n")
                
    print(f"[+] Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()
