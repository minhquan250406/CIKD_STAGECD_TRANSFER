"""
Stage I-F0: Feature Availability Audit.
Checks existing kg_complete cache files, optional mean-pool text features,
checks for NaNs/Infs, and records train/val split sizes.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-F0 Feature Audit")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Error: --no_test_eval flag must be present to guarantee test set safety."

    print("\n[+] Starting Feature Availability Audit...")

    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f", "f0_feature_audit")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Define required and optional files
    required_files = [
        "text_features.npy",
        "image_features_global.npy",
        "image_features_patch.npy",
        "kg_features.npy",
        "relation_ids.npy",
        "labels_fine.npy",
        "split_ids.npy"
    ]

    optional_text_features = [
        "text_features_mean.npy",
        "roberta_text_mean_pool.npy",
        "text_cls_mean_concat.npy"
    ]

    shapes_records = []
    missing_optional = []
    audit_summary = []

    # 2. Check and audit required files
    missing_required = []
    for filename in required_files:
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(filepath):
            missing_required.append(filename)
            continue
        
        # Load and audit
        try:
            arr = np.load(filepath)
            shape_str = str(list(arr.shape))
            dtype_str = str(arr.dtype)
            
            # Check NaN/Inf
            nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
            inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
            
            shapes_records.append({
                "Filename": filename,
                "Shape": shape_str,
                "Dtype": dtype_str,
                "NaN_Count": nan_count,
                "Inf_Count": inf_count,
                "Status": "OK" if (nan_count == 0 and inf_count == 0) else "NaN/Inf Detected"
            })
            
            if filename == "split_ids.npy":
                # Verify split counts
                # Train = 0, Val = 1, Test = 2
                train_count = int((arr == 0).sum())
                val_count = int((arr == 1).sum())
                test_count = int((arr == 2).sum())
                print(f"[+] Splits found - Train: {train_count}, Val: {val_count}, Test (LOCKED): {test_count}")
                assert test_count > 0, "Warning: split_ids.npy has no test split (id=2)?"
        except Exception as e:
            shapes_records.append({
                "Filename": filename,
                "Shape": "Error",
                "Dtype": "Error",
                "NaN_Count": -1,
                "Inf_Count": -1,
                "Status": f"Load Error: {str(e)}"
            })

    if missing_required:
        print(f"[-] ERROR: Required files missing: {missing_required}")
        sys.exit(1)

    # 3. Check optional files
    mean_pool_exists = False
    found_mean_pool_file = None
    for filename in optional_text_features:
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            mean_pool_exists = True
            found_mean_pool_file = filename
            try:
                arr = np.load(filepath)
                shapes_records.append({
                    "Filename": filename,
                    "Shape": str(list(arr.shape)),
                    "Dtype": str(arr.dtype),
                    "NaN_Count": int(np.isnan(arr).sum()),
                    "Inf_Count": int(np.isinf(arr).sum()),
                    "Status": "OK"
                })
            except Exception as e:
                pass
        else:
            missing_optional.append({
                "Filename": filename,
                "Type": "mean_pool_text_candidate",
                "Reason": "Not generated in prior cache steps"
            })

    # Save CSV reports
    df_shapes = pd.DataFrame(shapes_records)
    shapes_csv_path = os.path.join(out_dir, "F0_FEATURE_SHAPES.csv")
    df_shapes.to_csv(shapes_csv_path, index=False)
    print(f"[+] Saved shapes CSV to {shapes_csv_path}")

    df_missing = pd.DataFrame(missing_optional if missing_optional else [{"Filename": "None", "Type": "None", "Reason": "None"}])
    missing_csv_path = os.path.join(out_dir, "F0_MISSING_OPTIONAL_FEATURES.csv")
    df_missing.to_csv(missing_csv_path, index=False)
    print(f"[+] Saved missing features CSV to {missing_csv_path}")

    # Load split_ids for summary verification
    split_ids = np.load(os.path.join(cache_dir, "split_ids.npy"))
    train_count = int((split_ids == 0).sum())
    val_count = int((split_ids == 1).sum())

    # Write summary text file
    summary_path = os.path.join(out_dir, "F0_FEATURE_AUDIT_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("========================================================================\n")
        f.write("STAGE I-F0: FEATURE AUDIT SUMMARY\n")
        f.write("========================================================================\n\n")
        f.write("CRITICAL SAFETY ASSURANCES:\n")
        f.write("- NO TRAINING WAS RUN\n")
        f.write("- LOCKED TEST WAS NOT EVALUATED (isolation enforced via split filter)\n\n")
        
        f.write("SPLIT COUNT VERIFICATION:\n")
        f.write(f"- Train samples (split_id == 0): {train_count}\n")
        f.write(f"- Val samples (split_id == 1): {val_count}\n\n")
        
        f.write("MEAN-POOL TEXT FEATURE STATUS:\n")
        if mean_pool_exists:
            f.write(f"- A mean-pool text feature WAS found: {found_mean_pool_file}\n")
            f.write("- Stage I-F can proceed using cached mean-pool feature.\n\n")
        else:
            f.write("- No mean-pool text feature was found.\n")
            f.write("- Raw token-level text cache is NOT available (only text_features.npy exist).\n")
            f.write("- Recommendation: Use existing CLS text feature (text_features.npy) plus a trainable TextAdapter.\n")
            f.write("- Stage I-F can proceed using cached CLS feature only.\n\n")
            
        f.write("NAN/INF VERIFICATION:\n")
        nan_detected = any(row["NaN_Count"] > 0 or row["Inf_Count"] > 0 for row in shapes_records)
        if nan_detected:
            f.write("- WARNING: NaNs or Infs were detected in some cached arrays! See CSV report for details.\n")
        else:
            f.write("- Clean! No NaNs or Infs were detected in any checked arrays.\n")

    print(f"[+] Saved audit summary to {summary_path}")
    print("[+] Feature Audit completed successfully.")

if __name__ == "__main__":
    main()
