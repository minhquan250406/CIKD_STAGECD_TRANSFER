"""
Stage S1: Preparation Script.
Creates output directories, validates cached feature arrays, generates configurations,
and verifies train/val split sizes while strictly keeping the test set isolated.
"""

import os
import sys
import json
import argparse
import numpy as np

# Add workspace path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="Stage S1 CAFE-lite Preparation")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Error: --no_test_eval flag must be present to guarantee test set safety."

    print("\n[+] Starting Stage S1 CAFE-lite Preparation...")

    # 1. Directory paths setup
    config_dir = os.path.join(args.project_root, "configs", "stage_s1_cafe_lite")
    ckpt_dir = os.path.join(args.project_root, "checkpoints", "stage_s1_cafe_lite")
    out_dir = os.path.join(args.project_root, "outputs", "stage_s1_cafe_lite")
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Check and verify cached files exist
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    required_files = {
        "text_features.npy": [12786, 768],
        "image_features_global.npy": [12786, 512],
        "image_features_patch.npy": [12786, 49, 512],
        "labels_fine.npy": [12786],
        "split_ids.npy": [12786]
    }

    print("[+] Verifying cache files and shape expectations:")
    for fname, expected_shape in required_files.items():
        fpath = os.path.join(cache_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[-] ERROR: Cache file '{fname}' not found at {fpath}")
        
        arr = np.load(fpath, mmap_mode='r')
        actual_shape = list(arr.shape)
        print(f"    - {fname}: actual shape {actual_shape} | expected shape {expected_shape}")
        
        # Check alignment of sample counts (dim 0)
        assert actual_shape[0] == expected_shape[0], f"[-] Shape mismatch for {fname}: sample count {actual_shape[0]} vs {expected_shape[0]}"
        if len(expected_shape) > 1:
            assert actual_shape[1:] == expected_shape[1:], f"[-] Shape mismatch for {fname}: {actual_shape[1:]} vs {expected_shape[1:]}"

    # 3. Verify train/val split sizes
    split_ids = np.load(os.path.join(cache_dir, "split_ids.npy"))
    train_count = int(np.sum(split_ids == 0))
    val_count = int(np.sum(split_ids == 1))
    test_count = int(np.sum(split_ids == 2))

    print(f"[+] Loaded split counts:")
    print(f"    - Train (split_id==0): {train_count} (Expected: 8900)")
    print(f"    - Val (split_id==1):   {val_count} (Expected: 1300)")
    print(f"    - Test (split_id==2):  {test_count} (Expected: 2586)")

    assert train_count == 8900, f"[-] Train count mismatch: {train_count} vs 8900"
    assert val_count == 1300, f"[-] Val count mismatch: {val_count} vs 1300"
    assert test_count == 2586, f"[-] Test count mismatch: {test_count} vs 2586"

    # 4. Save configs
    configs = {
        "s1_cafe_lite_a_main": {
            "config_name": "s1_cafe_lite_a_main",
            "d_model": 256,
            "use_patch_pooling": False,
            "loss": "standard",
            "dropout": 0.1,
            "w_ambiguity_reg": 0.1,
            "batch_size": 16,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 20,
            "patience": 5,
            "seed": 42,
            "checkpoint_out": "checkpoints/stage_s1_cafe_lite/s1_cafe_lite_a_main.pt",
            "out_dir": "outputs/stage_s1_cafe_lite/s1_cafe_lite_a_main/"
        },
        "s1_cafe_lite_b_balanced": {
            "config_name": "s1_cafe_lite_b_balanced",
            "d_model": 256,
            "use_patch_pooling": False,
            "loss": "balanced",
            "dropout": 0.1,
            "w_ambiguity_reg": 0.1,
            "batch_size": 16,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 20,
            "patience": 5,
            "seed": 42,
            "checkpoint_out": "checkpoints/stage_s1_cafe_lite/s1_cafe_lite_b_balanced.pt",
            "out_dir": "outputs/stage_s1_cafe_lite/s1_cafe_lite_b_balanced/"
        },
        "s1_cafe_lite_c_patch_enabled": {
            "config_name": "s1_cafe_lite_c_patch_enabled",
            "d_model": 256,
            "use_patch_pooling": True,
            "loss": "balanced",
            "dropout": 0.1,
            "w_ambiguity_reg": 0.1,
            "batch_size": 16,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 20,
            "patience": 5,
            "seed": 42,
            "checkpoint_out": "checkpoints/stage_s1_cafe_lite/s1_cafe_lite_c_patch_enabled.pt",
            "out_dir": "outputs/stage_s1_cafe_lite/s1_cafe_lite_c_patch_enabled/"
        }
    }

    for cfg_name, payload in configs.items():
        cfg_path = os.path.join(config_dir, f"{cfg_name}.json")
        with open(cfg_path, "w") as f_cfg:
            json.dump(payload, f_cfg, indent=4)
        print(f"[+] Saved config file: {cfg_path}")

    # 5. Write S1_PREPARE_SUMMARY.txt
    summary_path = os.path.join(out_dir, "S1_PREPARE_SUMMARY.txt")
    with open(summary_path, "w") as f_sum:
        f_sum.write("========================================================================\n")
        f_sum.write("STAGE S1 PREPARATION SUMMARY\n")
        f_sum.write("========================================================================\n\n")
        f_sum.write("CRITICAL ASSURANCES:\n")
        f_sum.write("--------------------\n")
        f_sum.write("- NO TRAINING WAS RUN.\n")
        f_sum.write("- LOCKED TEST WAS NOT EVALUATED (isolation strictly enforced).\n")
        f_sum.write("- This is CAFE-lite / style-adapted, not official CAFE reproduction.\n")
        f_sum.write("- No KG/TVCS features were used.\n")
        f_sum.write("- Same kg_complete split is prepared for future validation-only training.\n\n")
        
        f_sum.write("CACHED FILES VERIFIED:\n")
        f_sum.write("----------------------\n")
        for fname, shape in required_files.items():
            f_sum.write(f"- {fname}: {shape} (OK)\n")
        f_sum.write("\n")
        
        f_sum.write("DATASET SPLIT COUNTS:\n")
        f_sum.write("---------------------\n")
        f_sum.write(f"- Train count: {train_count}\n")
        f_sum.write(f"- Val count:   {val_count}\n")
        f_sum.write(f"- Test count:  {test_count}\n\n")
        
        f_sum.write("CONFIGURATIONS GENERATED:\n")
        f_sum.write("-------------------------\n")
        for cfg_name in configs.keys():
            f_sum.write(f"- configs/stage_s1_cafe_lite/{cfg_name}.json\n")
            
        f_sum.write("\nPreparation completed successfully. Ready for dry-run checks.\n")

    print(f"[+] Saved preparation summary to: {summary_path}")
    print("[+] Stage S1 Preparation completed successfully.")

if __name__ == "__main__":
    main()
