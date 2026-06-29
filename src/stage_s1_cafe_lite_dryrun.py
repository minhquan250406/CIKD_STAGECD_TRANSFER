"""
Stage S1: CAFE-lite Dry-Run Script.
Executes a forward pass on a single validation batch under torch.no_grad(),
performs shape and sanity checks, computes numerical loss, and writes reports.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

# Add workspace path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage_s1_cafe_lite_model import CafeLiteSameSplitModel
from src.stage_s1_cafe_lite_losses import compute_cafe_lite_loss, check_tensor_sanity

def parse_args():
    parser = argparse.ArgumentParser(description="Stage S1 CAFE-lite Dry-Run")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Split to run dry-run on.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size to load.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Error: --no_test_eval flag must be present to guarantee test set safety."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Starting dry-run for config: {args.config}")
    print(f"[+] Device: {device}")

    # 1. Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"[-] Config file not found at: {args.config}")
    with open(args.config, "r") as f:
        config_data = json.load(f)
    
    config_name = config_data.get("config_name", os.path.basename(args.config))
    print(f"[+] Config '{config_name}' loaded successfully.")

    # 2. Check and load cache
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    split_ids = np.load(os.path.join(cache_dir, "split_ids.npy"))
    labels_fine = np.load(os.path.join(cache_dir, "labels_fine.npy"))
    text_features = np.load(os.path.join(cache_dir, "text_features.npy"))
    image_features_global = np.load(os.path.join(cache_dir, "image_features_global.npy"))
    image_features_patch = np.load(os.path.join(cache_dir, "image_features_patch.npy"))

    # Determine split indices and check safety
    if args.split == "val":
        split_val = 1
    elif args.split == "train":
        split_val = 0
    else:
        raise ValueError(f"Invalid split requested: {args.split}")

    split_mask = (split_ids == split_val)
    split_indices = np.where(split_mask)[0]
    
    if len(split_indices) < args.batch_size:
        raise ValueError(f"Requested batch size {args.batch_size} is larger than split size {len(split_indices)}")
    
    # Select the first B indices of the split
    batch_indices = split_indices[:args.batch_size]

    # Convert to PyTorch tensors and move to device
    bx_text = torch.tensor(text_features[batch_indices], dtype=torch.float32).to(device)
    bx_img_g = torch.tensor(image_features_global[batch_indices], dtype=torch.float32).to(device)
    bx_img_p = torch.tensor(image_features_patch[batch_indices], dtype=torch.float32).to(device)
    bx_lbl = torch.tensor(labels_fine[batch_indices], dtype=torch.long).to(device)

    # 3. Instantiate model
    model = CafeLiteSameSplitModel(
        d_model=config_data.get("d_model", 256),
        use_patch_pooling=config_data.get("use_patch_pooling", False),
        dropout=config_data.get("dropout", 0.1)
    ).to(device)
    model.eval()

    # 4. Forward pass
    print("[+] Executing forward pass (with no_grad, no backward)...")
    with torch.no_grad():
        logits, ambiguity_score, similarity_score, diagnostics = model(
            text_features=bx_text,
            image_features_global=bx_img_g,
            image_features_patch=bx_img_p if config_data.get("use_patch_pooling", False) else None
        )

    fused = diagnostics["fused"]

    # 5. Verify shapes
    print("[+] Verifying shape expectations:")
    print(f"    - logits:            {list(logits.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    - ambiguity_score:   {list(ambiguity_score.shape)} (Expected: [{args.batch_size}])")
    print(f"    - similarity_score:  {list(similarity_score.shape)} (Expected: [{args.batch_size}])")
    print(f"    - fused:             {list(fused.shape)} (Expected: [{args.batch_size}, {7 * config_data.get('d_model', 256)}])")

    assert list(logits.shape) == [args.batch_size, 6], "[-] Logits shape mismatch"
    assert list(ambiguity_score.shape) == [args.batch_size], "[-] Ambiguity score shape mismatch"
    assert list(similarity_score.shape) == [args.batch_size], "[-] Similarity score shape mismatch"
    
    expected_fused_dim = 7 * config_data.get("d_model", 256)
    assert list(fused.shape) == [args.batch_size, expected_fused_dim], "[-] Fused representation shape mismatch"

    # 6. Compute loss for numerical sanity check only
    # Calculate class weights and priors from training split only (split_ids == 0)
    train_mask = (split_ids == 0)
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_labels) / (6.0 * class_counts)
    
    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    loss_outputs = compute_cafe_lite_loss(
        logits=logits,
        targets=bx_lbl,
        ambiguity_score=ambiguity_score,
        similarity_score=similarity_score,
        loss_type=config_data.get("loss", "standard"),
        class_weights=class_weights_t,
        class_priors=class_priors_t,
        tau=0.5,
        w_ambiguity_reg=config_data.get("w_ambiguity_reg", 0.1)
    )

    loss_total = loss_outputs["loss_total"]
    loss_ce = loss_outputs["loss_ce"]
    loss_amb_reg = loss_outputs["loss_amb_reg"]

    print(f"[+] Computed losses:")
    print(f"    - Loss Total:         {loss_total.item():.6f}")
    print(f"    - Loss CE:            {loss_ce.item():.6f}")
    print(f"    - Loss Ambiguity Reg: {loss_amb_reg.item():.6f}")

    # 7. Check NaN/Inf and Sanity
    nan_count, inf_count = 0, 0
    tensors_to_check = {
        "logits": logits,
        "ambiguity_score": ambiguity_score,
        "similarity_score": similarity_score,
        "fused": fused,
        "loss_total": loss_total,
        "loss_ce": loss_ce,
        "loss_amb_reg": loss_amb_reg
    }

    for name, tensor in tensors_to_check.items():
        nc, ic = check_tensor_sanity(tensor, name)
        nan_count += nc
        inf_count += ic
        if nc > 0 or ic > 0:
            print(f"    [-] WARNING: {name} contains {nc} NaNs and {ic} Infs!")

    loss_is_finite = bool(torch.isfinite(loss_total).item())
    status = "PASSED" if (loss_is_finite and nan_count == 0 and inf_count == 0) else "FAILED"
    print(f"[+] Loss is finite: {loss_is_finite} | Status: {status}")

    # 8. Save report CSV (outputs/stage_s1_cafe_lite/S1_DRYRUN_REPORT.csv)
    out_dir = os.path.join(args.project_root, "outputs", "stage_s1_cafe_lite")
    report_csv = os.path.join(out_dir, "S1_DRYRUN_REPORT.csv")

    row_data = {
        "config_name": config_name,
        "split": args.split,
        "batch_size": args.batch_size,
        "logits_shape": str(list(logits.shape)),
        "ambiguity_shape": str(list(ambiguity_score.shape)),
        "similarity_shape": str(list(similarity_score.shape)),
        "fused_shape": str(list(fused.shape)),
        "loss_is_finite": loss_is_finite,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "loss_total": loss_total.item(),
        "loss_ce": loss_ce.item(),
        "loss_amb_reg": loss_amb_reg.item(),
        "status": status
    }

    if os.path.exists(report_csv):
        try:
            df = pd.read_csv(report_csv)
            # Remove existing row for this config to avoid duplication
            df = df[df["config_name"] != config_name]
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([row_data])
    else:
        df = pd.DataFrame([row_data])

    df.to_csv(report_csv, index=False)
    print(f"[+] Saved dry-run result row to: {report_csv}")

    # 9. Save/Append to outputs/stage_s1_cafe_lite/S1_DRYRUN_SUMMARY.txt
    summary_path = os.path.join(out_dir, "S1_DRYRUN_SUMMARY.txt")
    
    # We want to format the summary nicely and write it.
    with open(summary_path, "w") as f_sum:
        f_sum.write("========================================================================\n")
        f_sum.write("STAGE S1 DRY-RUN SUMMARY\n")
        f_sum.write("========================================================================\n\n")
        f_sum.write("CRITICAL ASSURANCES:\n")
        f_sum.write("--------------------\n")
        f_sum.write("- NO TRAINING WAS RUN.\n")
        f_sum.write("- LOCKED TEST WAS NOT EVALUATED (isolation strictly enforced).\n")
        f_sum.write("- This is CAFE-lite / style-adapted, not official CAFE reproduction.\n")
        f_sum.write("- No KG/TVCS features were used.\n")
        f_sum.write("- Same kg_complete split is prepared for future validation-only training.\n\n")
        
        f_sum.write("DRY-RUN EXECUTION DETAILS:\n")
        f_sum.write("--------------------------\n")
        f_sum.write(f"- Config name:        {config_name}\n")
        f_sum.write(f"- Split evaluated:    {args.split}\n")
        f_sum.write(f"- Batch size:         {args.batch_size}\n")
        f_sum.write(f"- Device used:        {device}\n")
        f_sum.write(f"- Logits shape:       {list(logits.shape)}\n")
        f_sum.write(f"- Ambiguity shape:    {list(ambiguity_score.shape)}\n")
        f_sum.write(f"- Similarity shape:   {list(similarity_score.shape)}\n")
        f_sum.write(f"- Fused shape:        {list(fused.shape)}\n\n")
        
        f_sum.write("NUMERICAL SANITY CHECKS:\n")
        f_sum.write("------------------------\n")
        f_sum.write(f"- Total Loss:         {loss_total.item():.6f}\n")
        f_sum.write(f"- CE Loss:            {loss_ce.item():.6f}\n")
        f_sum.write(f"- Ambiguity Reg Loss: {loss_amb_reg.item():.6f}\n")
        f_sum.write(f"- Loss is finite:     {loss_is_finite}\n")
        f_sum.write(f"- NaN count:          {nan_count}\n")
        f_sum.write(f"- Inf count:          {inf_count}\n")
        f_sum.write(f"- Dry-run status:     {status}\n\n")
        
        f_sum.write("Diagnostics values:\n")
        for k, v in diagnostics.items():
            if k != "fused":
                f_sum.write(f"  - {k}: {v}\n")
                
    print(f"[+] Saved dry-run summary to: {summary_path}")
    print("[+] Stage S1 Dry-Run completed successfully.")

if __name__ == "__main__":
    main()
