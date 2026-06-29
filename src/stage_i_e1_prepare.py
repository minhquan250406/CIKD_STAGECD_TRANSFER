"""
Stage I-E1: Preparation Script.
Creates configuration directories, writes JSON configs under configs/stage_i_e/,
and initializes output folders for future Stage I-E training stages.
"""

import os
import sys
import json
import argparse

# Add workspace path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-E1 Setup and Configuration.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Error: --no_test_eval flag must be present to guarantee test set safety."
    
    # 1. Paths setup
    config_dir = os.path.join(args.project_root, "configs", "stage_i_e")
    ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_i_e")
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e")
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # Load class mask to find bottleneck classes
    mask_path = os.path.join(out_dir, "e0_bottleneck_audit", "E0_RECOMMENDED_CLASS_MASK.json")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Recommended class mask not found: {mask_path}. Please run E0 bottleneck audit first.")
        
    with open(mask_path, "r") as f:
        mask_data = json.load(f)
        
    class_mask = mask_data["class_mask"]
    # Bottleneck classes: mask is 1.0 (lowest F1 performing minority classes)
    bottleneck_classes = [i for i, v in enumerate(class_mask) if v == 1.0]
    print(f"[+] Loaded class mask: {class_mask}")
    print(f"[+] Identified bottleneck classes: {bottleneck_classes}")
    
    # 2. Write configs for beta = 0.1, 0.2, 0.3
    betas = [0.1, 0.2, 0.3]
    config_files = []
    
    for beta in betas:
        beta_str = f"{beta:.1f}".replace('.', '')
        config_name = f"i_e1_beta{beta_str}"
        
        config_payload = {
            "config_name": config_name,
            "beta": beta,
            "class_mask": class_mask,
            "bottleneck_classes": bottleneck_classes,
            "class_mask_path": mask_path,
            
            # Loss weights
            "w_balanced": 1.00,
            "w_focal": 0.30,
            "w_ck_guard": 0.10,
            "w_kl": 0.08,
            "w_reg": 0.02,
            
            # Hyperparameters
            "tau_logit_adjust": 0.5,
            "focal_gamma": 1.5,
            "kl_temperature": 1.0,
            "hidden_dim": 128,
            "dropout": 0.1,
            
            # Training hyperparameters (placeholders for future training)
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 10,
            "patience": 3,
            "batch_size": 16,
            
            # Paths
            "tvcs_checkpoint": "checkpoints/stage_f/tvcs_specialist_seed42_padded_for_f2.pt",
            "f4_checkpoint": "outputs/stage_f3_ablation/no_c_emb/cikd_pp_rt_ablation_no_c_emb.pt",
            "checkpoint_out": f"checkpoints/stage_i_e/{config_name}.pt",
            "out_dir": f"outputs/stage_i_macro_micro_improvement/stage_i_e/{config_name}/"
        }
        
        cfg_file_path = os.path.join(config_dir, f"{config_name}.json")
        with open(cfg_file_path, "w") as f_cfg:
            json.dump(config_payload, f_cfg, indent=4)
        config_files.append(cfg_file_path)
        print(f"[+] Saved config file: {cfg_file_path}")
        
    # 3. Save a summary of preparation
    summary_path = os.path.join(out_dir, "E1_PREPARE_SUMMARY.txt")
    with open(summary_path, "w") as f_sum:
        f_sum.write("========================================================================\n")
        f_sum.write("STAGE I-E1 PREPARATION SUMMARY\n")
        f_sum.write("========================================================================\n\n")
        f_sum.write("CRITICAL ASSURANCES:\n")
        f_sum.write("--------------------\n")
        f_sum.write("- NO TRAINING RUN OR EVALUATION EXECUTED.\n")
        f_sum.write("- TEST SET LOCKED ISOLATION IS ENFORCED.\n\n")
        
        f_sum.write("CONFIGURATIONS GENERATED:\n")
        f_sum.write("-------------------------\n")
        for beta, cfg_p in zip(betas, config_files):
            f_sum.write(f"- Beta: {beta:.1f} | Config Path: configs/stage_i_e/{os.path.basename(cfg_p)}\n")
            
        f_sum.write("\nPending execution of dry-run script to check forward passes and shapes.\n")
        
    print(f"[+] Saved preparation summary to: {summary_path}")
    print("[+] Setup completed successfully.")

if __name__ == "__main__":
    main()
