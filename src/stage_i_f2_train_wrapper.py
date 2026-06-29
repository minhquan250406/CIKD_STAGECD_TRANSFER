"""
Stage I-F2: Safe Refinement Training Wrapper.
Invokes the existing stage_i_f1_train.py runner with Stage I-F2 configurations.
This script does not run automatically during the preparation phase.
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper to train Stage I-F2 configurations.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the Stage I-F2 JSON config file.")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate override.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to ensure test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Abort: --no_test_eval flag is missing! Must be present to ensure test set isolation."
    
    # Verify that the config is indeed an F2 refinement config
    config_name = os.path.basename(args.config)
    if "i_f2_" not in config_name:
        print(f"ERROR: Configuration {config_name} does not appear to be a Stage I-F2 refinement configuration.")
        sys.exit(1)
         
    # Build command to run stage_i_f1_train.py
    cmd = [
        sys.executable,
        os.path.join(args.project_root, "src", "stage_i_f1_train.py"),
        "--project_root", args.project_root,
        "--config", args.config,
        "--max_epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--weight_decay", str(args.weight_decay),
        "--seed", str(args.seed),
        "--no_test_eval"
    ]
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
        
    print(f"[+] Launching training runner with command: {' '.join(cmd)}")
    
    # Run command
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
