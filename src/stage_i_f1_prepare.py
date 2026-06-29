"""
Stage I-F1: Preparation Script.
Creates output directories, generates config files, and verifies critical artifacts.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# Add workspace path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-F1 Preparation")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Error: --no_test_eval flag must be present to guarantee test set safety."

    print("\n[+] Starting Stage I-F Preparation...")

    # 1. Directory paths setup
    config_dir = os.path.join(args.project_root, "configs", "stage_i_f")
    ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_i_f")
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f")
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Check and verify required artifacts exist
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    baseline_dir = os.path.join(args.project_root, "outputs", "stage_f0_baseline_anchor")
    f4_checkpoint = os.path.join(args.project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
    tvcs_checkpoint = os.path.join(args.project_root, "checkpoints", "stage_f", "tvcs_specialist_seed42_padded_for_f2.pt")
    class_mask_path = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e", "e0_bottleneck_audit", "E0_RECOMMENDED_CLASS_MASK.json")

    paths_to_check = {
        "kg_complete cache dir": cache_dir,
        "baseline logits dir": baseline_dir,
        "F4 checkpoint": f4_checkpoint,
        "TVCS checkpoint": tvcs_checkpoint,
        "Class mask JSON": class_mask_path
    }

    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"[-] ERROR: Required artifact '{name}' not found at: {path}")
        print(f"[+] Verified existence: {name} -> {path}")

    # Load class mask
    with open(class_mask_path, "r") as f:
        mask_data = json.load(f)
    class_mask = mask_data["class_mask"]
    print(f"[+] Loaded class mask: {class_mask}")

    # 3. Verify checkpoints can be loaded safely
    try:
        f4_state = torch.load(f4_checkpoint, map_location="cpu", weights_only=False)
        print(f"[+] Verified F4 checkpoint loads successfully. Keys: {list(f4_state.keys()) if isinstance(f4_state, dict) else 'non-dict'}")
    except Exception as e:
        raise RuntimeError(f"[-] ERROR: Failed to load F4 checkpoint: {str(e)}")

    try:
        tvcs_state = torch.load(tvcs_checkpoint, map_location="cpu", weights_only=False)
        print(f"[+] Verified TVCS checkpoint loads successfully. Keys: {list(tvcs_state.keys()) if isinstance(tvcs_state, dict) else 'non-dict'}")
    except Exception as e:
        raise RuntimeError(f"[-] ERROR: Failed to load TVCS checkpoint: {str(e)}")

    # 4. Verify train/val splits only
    split_ids_path = os.path.join(cache_dir, "split_ids.npy")
    split_ids = np.load(split_ids_path)
    train_indices = np.where(split_ids == 0)[0]
    val_indices = np.where(split_ids == 1)[0]
    print(f"[+] Train count: {len(train_indices)}, Val count: {len(val_indices)}")
    
    # 5. Generate configuration payloads
    configs = {
        "i_f_a_gamma02_main": {
            "config_name": "i_f_a_gamma02_main",
            "gamma": 0.2,
            "use_patch_adapter": True,
            "use_kg_relation_adapter": True,
            "use_tvcs_zv": True,
            "loss_focus_classes": [1, 2, 5],
            
            # Loss weights
            "w_balanced": 1.00,
            "w_focal": 0.20,
            "w_ck_guard": 0.10,
            "w_kl": 0.10,
            "w_delta_norm": 0.03,
            
            # Hyperparameters
            "tau_logit_adjust": 0.5,
            "focal_gamma": 1.5,
            "kl_temperature": 1.0,
            "d_model": 256,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            
            # Training parameters (placeholders)
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 10,
            "patience": 3,
            "batch_size": 16,
            
            # Paths
            "f4_checkpoint": "outputs/stage_f3_ablation/no_c_emb/cikd_pp_rt_ablation_no_c_emb.pt",
            "tvcs_checkpoint": "checkpoints/stage_f/tvcs_specialist_seed42_padded_for_f2.pt",
            "checkpoint_out": "checkpoints/stage_i_f/i_f_a_gamma02_main.pt",
            "out_dir": "outputs/stage_i_macro_micro_improvement/stage_i_f/i_f_a_gamma02_main/"
        },
        "i_f_b_gamma01_safe": {
            "config_name": "i_f_b_gamma01_safe",
            "gamma": 0.1,
            "stronger_anchor": True,
            "use_patch_adapter": True,
            "use_kg_relation_adapter": True,
            "use_tvcs_zv": True,
            "loss_focus_classes": [1, 2, 5],
            
            # Loss weights
            "w_balanced": 1.00,
            "w_focal": 0.20,
            "w_ck_guard": 0.10,
            "w_kl": 0.15,
            "w_delta_norm": 0.05,
            
            # Hyperparameters
            "tau_logit_adjust": 0.5,
            "focal_gamma": 1.5,
            "kl_temperature": 1.0,
            "d_model": 256,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            
            # Training parameters (placeholders)
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 10,
            "patience": 3,
            "batch_size": 16,
            
            # Paths
            "f4_checkpoint": "outputs/stage_f3_ablation/no_c_emb/cikd_pp_rt_ablation_no_c_emb.pt",
            "tvcs_checkpoint": "checkpoints/stage_f/tvcs_specialist_seed42_padded_for_f2.pt",
            "checkpoint_out": "checkpoints/stage_i_f/i_f_b_gamma01_safe.pt",
            "out_dir": "outputs/stage_i_macro_micro_improvement/stage_i_f/i_f_b_gamma01_safe/"
        },
        "i_f_c_gamma03_strong": {
            "config_name": "i_f_c_gamma03_strong",
            "gamma": 0.3,
            "use_patch_adapter": True,
            "use_kg_relation_adapter": True,
            "use_tvcs_zv": True,
            "loss_focus_classes": [1, 2, 5],
            
            # Loss weights
            "w_balanced": 1.00,
            "w_focal": 0.20,
            "w_ck_guard": 0.10,
            "w_kl": 0.10,
            "w_delta_norm": 0.03,
            
            # Hyperparameters
            "tau_logit_adjust": 0.5,
            "focal_gamma": 1.5,
            "kl_temperature": 1.0,
            "d_model": 256,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            
            # Training parameters (placeholders)
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 10,
            "patience": 3,
            "batch_size": 16,
            
            # Paths
            "f4_checkpoint": "outputs/stage_f3_ablation/no_c_emb/cikd_pp_rt_ablation_no_c_emb.pt",
            "tvcs_checkpoint": "checkpoints/stage_f/tvcs_specialist_seed42_padded_for_f2.pt",
            "checkpoint_out": "checkpoints/stage_i_f/i_f_c_gamma03_strong.pt",
            "out_dir": "outputs/stage_i_macro_micro_improvement/stage_i_f/i_f_c_gamma03_strong/"
        },
        "i_f_d_cls_only_ablation": {
            "config_name": "i_f_d_cls_only_ablation",
            "gamma": 0.2,
            "use_patch_adapter": False,
            "use_text_cls_only": True,
            "use_kg_relation_adapter": True,
            "use_tvcs_zv": True,
            "loss_focus_classes": [1, 2, 5],
            
            # Loss weights
            "w_balanced": 1.00,
            "w_focal": 0.20,
            "w_ck_guard": 0.10,
            "w_kl": 0.10,
            "w_delta_norm": 0.03,
            
            # Hyperparameters
            "tau_logit_adjust": 0.5,
            "focal_gamma": 1.5,
            "kl_temperature": 1.0,
            "d_model": 256,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            
            # Training parameters (placeholders)
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 10,
            "patience": 3,
            "batch_size": 16,
            
            # Paths
            "f4_checkpoint": "outputs/stage_f3_ablation/no_c_emb/cikd_pp_rt_ablation_no_c_emb.pt",
            "tvcs_checkpoint": "checkpoints/stage_f/tvcs_specialist_seed42_padded_for_f2.pt",
            "checkpoint_out": "checkpoints/stage_i_f/i_f_d_cls_only_ablation.pt",
            "out_dir": "outputs/stage_i_macro_micro_improvement/stage_i_f/i_f_d_cls_only_ablation/"
        }
    }

    # Write configs
    for filename, payload in configs.items():
        cfg_path = os.path.join(config_dir, f"{filename}.json")
        with open(cfg_path, "w") as f_cfg:
            json.dump(payload, f_cfg, indent=4)
        print(f"[+] Saved config: {cfg_path}")

    # 6. Save summary report
    summary_path = os.path.join(out_dir, "F1_PREPARE_SUMMARY.txt")
    with open(summary_path, "w") as f_sum:
        f_sum.write("========================================================================\n")
        f_sum.write("STAGE I-F1 PREPARATION SUMMARY\n")
        f_sum.write("========================================================================\n\n")
        f_sum.write("CRITICAL ASSURANCES:\n")
        f_sum.write("--------------------\n")
        f_sum.write("- NO TRAINING WAS RUN.\n")
        f_sum.write("- LOCKED TEST WAS NOT EVALUATED (isolation enforced).\n\n")
        
        f_sum.write("REQUIRED CHECKPOINTS VERIFIED:\n")
        f_sum.write(f"- F4 checkpoint (no_c_emb): {f4_checkpoint} (OK)\n")
        f_sum.write(f"- TVCS specialist checkpoint: {tvcs_checkpoint} (OK)\n\n")
        
        f_sum.write("CONFIGURATIONS GENERATED:\n")
        f_sum.write("-------------------------\n")
        for filename in configs.keys():
            f_sum.write(f"- configs/stage_i_f/{filename}.json\n")
            
        f_sum.write("\nPending dry-run validation checks.\n")

    print(f"[+] Saved preparation summary to: {summary_path}")
    print("[+] Stage I-F Preparation completed successfully.")

if __name__ == "__main__":
    main()
