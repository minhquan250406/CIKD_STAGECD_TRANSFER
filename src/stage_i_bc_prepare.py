"""
Stage I-BC: Preparation and Scaffold Setup script.
Verifies all cached input shapes, checks baseline path availability,
loads checkpoint metadata, generates 4 configuration JSONs, and exports
the I_BC_PREPARE_SUMMARY.txt file.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-BC Prepare configs and directory scaffold.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    return parser.parse_args()

def main():
    args = parse_args()
    project_root = args.project_root
    
    # 1. Create output folders
    out_dir = os.path.join(project_root, "outputs", "stage_i_macro_micro_improvement")
    ckpt_dir = os.path.join(project_root, "checkpoints", "stage_i")
    config_dir = os.path.join(project_root, "configs", "stage_i_bc")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    # 2. Check cache paths & shapes
    cache_dir = os.path.join(project_root, "data", "cache", "kg_complete")
    cache_files = {
        'text_features.npy': (12786, 768),
        'image_features_global.npy': (12786, 512),
        'image_features_patch.npy': (12786, 49, 512),
        'kg_features.npy': (12786, 100),
        'relation_ids.npy': (12786,),
        'labels_fine.npy': (12786,),
        'y_ck.npy': (12786,),
        'split_ids.npy': (12786,)
    }
    
    cache_summary = []
    cache_missing = []
    for fname, expected_shape in cache_files.items():
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath):
            arr = np.load(fpath)
            shape = arr.shape
            cache_summary.append(f"{fname}: shape {shape} (expected {expected_shape})")
            if len(shape) != len(expected_shape) or any(shape[i] != expected_shape[i] for i in range(len(shape))):
                cache_summary.append(f"  WARNING: shape mismatch for {fname}")
        else:
            cache_missing.append(fname)
            cache_summary.append(f"{fname}: MISSING")
            
    # 3. Check baseline logits path availability
    baseline_dir = os.path.join(project_root, "outputs", "stage_f0_baseline_anchor")
    baseline_files = ['train_logits_base.npy', 'val_logits_base.npy', 'test_logits_base.npy']
    baseline_status = []
    for fname in baseline_files:
        fpath = os.path.join(baseline_dir, fname)
        if os.path.exists(fpath):
            arr = np.load(fpath)
            baseline_status.append(f"{fname}: shape {arr.shape}")
        else:
            baseline_status.append(f"{fname}: MISSING")
            
    # 4. Load F4 checkpoint metadata if possible
    f4_ckpt_path = os.path.join(project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
    f4_metadata = "Not found"
    if os.path.exists(f4_ckpt_path):
        try:
            ckpt = torch.load(f4_ckpt_path, map_location='cpu', weights_only=False)
            keys = list(ckpt.keys())
            f4_metadata = f"Found. Keys: {keys}"
            if 'epoch' in ckpt:
                f4_metadata += f", Best Epoch: {ckpt['epoch']}"
            if 'val_metrics' in ckpt:
                # Truncate large output lists to prevent messiness
                metrics = {k: (v if not isinstance(v, list) else f"list of len {len(v)}") for k, v in ckpt['val_metrics'].items()}
                f4_metadata += f", Val Metrics: {metrics}"
        except Exception as e:
            f4_metadata = f"Found but error loading: {e}"
            
    # 5. Load TVCS checkpoint metadata if possible
    tvcs_ckpt_path = os.path.join(project_root, "checkpoints", "stage_f", "tvcs_specialist_seed42_padded_for_f2.pt")
    tvcs_metadata = "Not found"
    if os.path.exists(tvcs_ckpt_path):
        try:
            ckpt = torch.load(tvcs_ckpt_path, map_location='cpu', weights_only=False)
            keys = list(ckpt.keys())
            tvcs_metadata = f"Found. Keys: {keys}"
            if 'epoch' in ckpt:
                tvcs_metadata += f", Epoch: {ckpt['epoch']}"
            if 'val_metrics' in ckpt:
                metrics = {k: (v if not isinstance(v, list) else f"list of len {len(v)}") for k, v in ckpt['val_metrics'].items()}
                tvcs_metadata += f", Val Metrics: {metrics}"
        except Exception as e:
            tvcs_metadata = f"Found but error loading: {e}"
            
    # 6. Save config JSON files
    configs = {
        "i_bc_a.json": {
            "config_name": "I-BC-A",
            "alpha_max": 0.5,
            "tau_logit_adjust": 0.5,
            "binary_loss_weight": 0.25,
            "ck_real_loss_weight": 0.35,
            "kl_anchor_weight": 0.05,
            "residual_reg_weight": 0.01,
            "epochs": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "patience": 5,
            "dropout": 0.2,
            "seed": 42
        },
        "i_bc_b.json": {
            "config_name": "I-BC-B",
            "alpha_max": 0.5,
            "tau_logit_adjust": 1.0,
            "binary_loss_weight": 0.25,
            "ck_real_loss_weight": 0.35,
            "kl_anchor_weight": 0.05,
            "residual_reg_weight": 0.01,
            "epochs": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "patience": 5,
            "dropout": 0.2,
            "seed": 42
        },
        "i_bc_c.json": {
            "config_name": "I-BC-C",
            "alpha_max": 0.3,
            "tau_logit_adjust": 0.5,
            "binary_loss_weight": 0.25,
            "ck_real_loss_weight": 0.50,
            "kl_anchor_weight": 0.05,
            "residual_reg_weight": 0.01,
            "epochs": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "patience": 5,
            "dropout": 0.2,
            "seed": 42
        },
        "i_bc_d.json": {
            "config_name": "I-BC-D",
            "alpha_max": 0.5,
            "tau_logit_adjust": 0.5,
            "binary_loss_weight": 0.15,
            "ck_real_loss_weight": 0.50,
            "kl_anchor_weight": 0.05,
            "residual_reg_weight": 0.01,
            "epochs": 20,
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "patience": 5,
            "dropout": 0.2,
            "seed": 42
        }
    }
    
    for cname, cdata in configs.items():
        cpath = os.path.join(config_dir, cname)
        with open(cpath, 'w') as f:
            json.dump(cdata, f, indent=4)
            
    # Write I_BC_PREPARE_SUMMARY.txt
    summary_path = os.path.join(out_dir, "I_BC_PREPARE_SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write("STAGE I-BC PREPARATION SUMMARY\n")
        f.write("==============================\n\n")
        f.write("1. FILES CREATED:\n")
        f.write("   - src/stage_i_bc_multitask_balanced_model.py\n")
        f.write("   - src/stage_i_bc_losses.py\n")
        f.write("   - src/stage_i_bc_prepare.py\n")
        f.write("   - src/stage_i_bc_dryrun.py\n")
        for cname in configs.keys():
            f.write(f"   - configs/stage_i_bc/{cname}\n")
        f.write("\n2. DIRECTORIES CREATED:\n")
        f.write(f"   - {out_dir}\n")
        f.write(f"   - {ckpt_dir}\n")
        f.write(f"   - {config_dir}\n\n")
        f.write("3. EXISTING ARTIFACTS AND STATUS:\n")
        f.write(f"   - Cache complete path: {cache_dir}\n")
        f.write(f"     Missing cache files: {cache_missing if cache_missing else 'None'}\n")
        f.write(f"   - F4 checkpoint path: {f4_ckpt_path}\n")
        f.write(f"     F4 metadata: {f4_metadata}\n")
        f.write(f"   - TVCS checkpoint path: {tvcs_ckpt_path}\n")
        f.write(f"     TVCS metadata: {tvcs_metadata}\n\n")
        f.write("4. CACHE SHAPE SUMMARY:\n")
        for s in cache_summary:
            f.write(f"  - {s}\n")
        f.write("\n5. BASELINE LOGITS STATUS:\n")
        for s in baseline_status:
            f.write(f"  - {s}\n")
        f.write("\n6. CONFIG SUMMARY:\n")
        for cname, cdata in configs.items():
            f.write(f"  - {cname} ({cdata['config_name']}):\n")
            f.write(f"    alpha_max:           {cdata['alpha_max']}\n")
            f.write(f"    tau_logit_adjust:    {cdata['tau_logit_adjust']}\n")
            f.write(f"    binary_loss_weight:  {cdata['binary_loss_weight']}\n")
            f.write(f"    ck_real_loss_weight: {cdata['ck_real_loss_weight']}\n")
            f.write(f"    kl_anchor_weight:    {cdata['kl_anchor_weight']}\n")
            f.write(f"    residual_reg_weight: {cdata['residual_reg_weight']}\n")
        f.write("\n7. DRY-RUN STATUS:\n")
        f.write("   - Configured and ready. Pending dry-run execution checks.\n\n")
        f.write("8. CRITICAL VERIFICATION STATEMENTS:\n")
        f.write("   - NO TRAINING WAS RUN\n")
        f.write("   - LOCKED TEST WAS NOT EVALUATED\n")
        
    print(f"[+] Saved preparation summary to: {summary_path}")

if __name__ == "__main__":
    main()
