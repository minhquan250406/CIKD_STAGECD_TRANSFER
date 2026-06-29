"""
Stage I-F2: Safe Refinement Preparation.
Sets up directories, saves 5 configuration files under configs/stage_i_f/f2_refinement/,
and verifies all training artifacts and dataset dependencies.
"""

import os
import sys
import json
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Stage I-F2 configurations and folders.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Enforce safety gate: must be present to ensure test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Abort: --no_test_eval flag is missing! Must be present to ensure test set isolation."

    print("========================================================================")
    print("STAGE I-F2: SAFE REFINEMENT PREPARATION")
    print("========================================================================")
    print(f"Project root: {args.project_root}")

    # 1. Define folders to create
    f2_out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f", "f2_refinement")
    f2_ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_i_f", "f2_refinement")
    f2_config_dir = os.path.join(args.project_root, "configs", "stage_i_f", "f2_refinement")

    print("[+] Creating Stage I-F2 directories...")
    os.makedirs(f2_out_dir, exist_ok=True)
    os.makedirs(f2_ckpt_dir, exist_ok=True)
    os.makedirs(f2_config_dir, exist_ok=True)
    print(f"    - Output dir: {f2_out_dir}")
    print(f"    - Checkpoint dir: {f2_ckpt_dir}")
    print(f"    - Config dir: {f2_config_dir}")

    # 2. Define configurations
    configs = {
        "i_f2_a_gamma008_lr5e5_kl015_l205": {
            "gamma": 0.08,
            "lr": 5e-5,
            "w_kl": 0.15,
            "w_delta_norm": 0.05,
            "w_focal": 0.20,
            "w_ck_guard": 0.10
        },
        "i_f2_b_gamma010_lr5e5_kl020_l208": {
            "gamma": 0.10,
            "lr": 5e-5,
            "w_kl": 0.20,
            "w_delta_norm": 0.08,
            "w_focal": 0.20,
            "w_ck_guard": 0.10
        },
        "i_f2_c_gamma012_lr5e5_kl015_l205": {
            "gamma": 0.12,
            "lr": 5e-5,
            "w_kl": 0.15,
            "w_delta_norm": 0.05,
            "w_focal": 0.20,
            "w_ck_guard": 0.10
        },
        "i_f2_d_gamma010_lr1e4_kl020_l208": {
            "gamma": 0.10,
            "lr": 1e-4,
            "w_kl": 0.20,
            "w_delta_norm": 0.08,
            "w_focal": 0.20,
            "w_ck_guard": 0.10
        },
        "i_f2_e_gamma005_lr5e5_kl020_l210": {
            "gamma": 0.05,
            "lr": 5e-5,
            "w_kl": 0.20,
            "w_delta_norm": 0.10,
            "w_focal": 0.20,
            "w_ck_guard": 0.10
        }
    }

    base_config = {
        "stronger_anchor": True,
        "use_patch_adapter": True,
        "use_kg_relation_adapter": True,
        "use_tvcs_zv": True,
        "loss_focus_classes": [1, 2, 5],
        "w_balanced": 1.0,
        "tau_logit_adjust": 0.5,
        "focal_gamma": 1.5,
        "kl_temperature": 1.0,
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "weight_decay": 0.0001,
        "epochs": 10,
        "patience": 3,
        "batch_size": 16,
        "f4_checkpoint": "outputs/stage_f3_ablation/no_c_emb/cikd_pp_rt_ablation_no_c_emb.pt",
        "tvcs_checkpoint": "checkpoints/stage_f/tvcs_specialist_seed42_padded_for_f2.pt"
    }

    print("[+] Saving configurations...")
    saved_paths = []
    for name, overrides in configs.items():
        cfg = base_config.copy()
        cfg["config_name"] = name
        cfg.update(overrides)
        cfg["checkpoint_out"] = f"checkpoints/stage_i_f/f2_refinement/{name}.pt"
        cfg["out_dir"] = f"outputs/stage_i_macro_micro_improvement/stage_i_f/f2_refinement/{name}/"
        
        cfg_path = os.path.join(f2_config_dir, f"{name}.json")
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)
        print(f"    - Saved: {cfg_path}")
        saved_paths.append(cfg_path)

    # 3. Verification checks
    print("[+] Performing verification checks...")
    
    # 3a. Verify F1 scripts exist
    f1_scripts = [
        "src/stage_i_f1_feature_refresh_model.py",
        "src/stage_i_f1_losses.py",
        "src/stage_i_f1_train.py"
    ]
    f1_scripts_status = {}
    for script in f1_scripts:
        path = os.path.join(args.project_root, script)
        exists = os.path.exists(path)
        f1_scripts_status[script] = "EXISTS" if exists else "MISSING"
        print(f"    - {script}: {f1_scripts_status[script]}")
        assert exists, f"Prerequisite script missing: {script}"

    # 3b. Verify F4 checkpoint exists
    f4_ckpt = os.path.join(args.project_root, base_config["f4_checkpoint"])
    f4_exists = os.path.exists(f4_ckpt)
    print(f"    - F4 Backbone Checkpoint ({base_config['f4_checkpoint']}): {'EXISTS' if f4_exists else 'MISSING'}")
    assert f4_exists, f"F4 Backbone checkpoint missing at: {f4_ckpt}"

    # 3c. Verify TVCS checkpoint exists
    tvcs_ckpt = os.path.join(args.project_root, base_config["tvcs_checkpoint"])
    tvcs_exists = os.path.exists(tvcs_ckpt)
    print(f"    - TVCS Specialist Checkpoint ({base_config['tvcs_checkpoint']}): {'EXISTS' if tvcs_exists else 'MISSING'}")
    assert tvcs_exists, f"TVCS Specialist checkpoint missing at: {tvcs_ckpt}"

    # 3d. Verify kg_complete cache shape
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    print(f"    - Checking dataset cache shape at: {cache_dir}")
    assert os.path.exists(cache_dir), f"Dataset cache directory missing: {cache_dir}"
    
    cache_files = {
        'split_ids.npy': None,
        'relation_ids.npy': None,
        'kg_features.npy': None,
        'labels_fine.npy': None,
        'text_features.npy': None,
        'image_features_global.npy': None,
        'image_features_patch.npy': None
    }
    
    for filename in cache_files:
        path = os.path.join(cache_dir, filename)
        assert os.path.exists(path), f"Cache file missing: {filename}"
        arr = np.load(path)
        cache_files[filename] = arr.shape
        print(f"      * {filename}: shape {arr.shape}")

    # 3e. Verify F1 best summary and best checkpoint exist
    f1_summary_path = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f", "f1_training", "F1_TRAINING_SUMMARY.txt")
    f1_summary_exists = os.path.exists(f1_summary_path)
    print(f"    - F1 Training Summary: {'EXISTS' if f1_summary_exists else 'MISSING'}")
    assert f1_summary_exists, f"F1 Training Summary missing at: {f1_summary_path}"

    f1_best_ckpt_path = os.path.join(args.project_root, "checkpoints", "stage_i_f", "best_i_f_b_gamma01_safe.pt")
    f1_best_ckpt_exists = os.path.exists(f1_best_ckpt_path)
    print(f"    - F1 Best Checkpoint (best_i_f_b_gamma01_safe.pt): {'EXISTS' if f1_best_ckpt_exists else 'MISSING'}")
    assert f1_best_ckpt_exists, f"F1 Best Checkpoint missing at: {f1_best_ckpt_path}"

    # 4. Save prepare summary
    summary_out_path = os.path.join(f2_out_dir, "F2_PREPARE_SUMMARY.txt")
    print(f"[+] Saving preparation summary to: {summary_out_path}")
    
    with open(summary_out_path, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE I-F2: SAFE REFINEMENT PREPARATION SUMMARY\n")
        f.write("========================================================================\n\n")
        f.write(f"Project Root: {args.project_root}\n")
        f.write("Status: SUCCESS\n\n")
        
        f.write("CREATED DIRECTORIES:\n")
        f.write(f"  - Output dir: {f2_out_dir}\n")
        f.write(f"  - Checkpoint dir: {f2_ckpt_dir}\n")
        f.write(f"  - Config dir: {f2_config_dir}\n\n")
        
        f.write("GENERATED STAGE I-F2 CONFIGURATIONS:\n")
        for name, overrides in configs.items():
            f.write(f"  - {name}:\n")
            for k, v in overrides.items():
                f.write(f"      {k}: {v}\n")
            f.write(f"      checkpoint_out: checkpoints/stage_i_f/f2_refinement/{name}.pt\n")
            f.write(f"      out_dir: outputs/stage_i_macro_micro_improvement/stage_i_f/f2_refinement/{name}/\n")
            
        f.write("\nVERIFIED SCRIPTS & BACKBONES:\n")
        for script, status in f1_scripts_status.items():
            f.write(f"  - {script}: {status}\n")
        f.write(f"  - F4 Backbone checkpoint ({base_config['f4_checkpoint']}): EXISTS\n")
        f.write(f"  - TVCS Specialist checkpoint ({base_config['tvcs_checkpoint']}): EXISTS\n")
        f.write(f"  - F1 Best Checkpoint (best_i_f_b_gamma01_safe.pt): EXISTS\n")
        f.write(f"  - F1 Training Summary (F1_TRAINING_SUMMARY.txt): EXISTS\n\n")
        
        f.write("DATASET CACHE SHAPES:\n")
        for filename, shape in cache_files.items():
            f.write(f"  - {filename}: {shape}\n")
            
    print("[+] Stage I-F2 Preparation Complete.")

if __name__ == "__main__":
    main()
