"""
Stage I-E1: Dry-Run Verification script.
Executes a single forward pass under no_grad to verify shapes and loss computation,
runs NaN/Inf checks, and writes results to E1_DRYRUN_REPORT.csv.
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

from src.models.cikd_pp_rt import CIKDPPResidualTransformer
from src.stage_i_e1_bottleneck_adapter_model import StageIE1BottleneckAdapterModel
from src.stage_i_e1_losses import compute_adapter_total_loss, check_tensor_sanity

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-E1 Adapter Dry-Run Verification.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON config file.")
    parser.add_argument("--split", type=str, default="val", choices=["val"],
                        help="Split to use. Strictly restricted to validation split.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for dry-run forward pass.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Safety gate: must be present to enforce test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.split == "val", "Error: Dry-run is strictly restricted to validation split."
    assert args.no_test_eval, "Error: --no_test_eval flag must be present to guarantee test set safety."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Running dry-run for config: {args.config} on split: {args.split}")
    
    # 1. Load config file
    if not os.path.exists(args.config):
        print(f"[-] Config file not found: {args.config}")
        sys.exit(1)
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    config_name = config_data.get("config_name", os.path.basename(args.config))
    
    # 2. Check cache and load validation logits
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    baseline_dir = os.path.join(args.project_root, "outputs", "stage_f0_baseline_anchor")
    
    split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
    relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
    kg_features = np.load(os.path.join(cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
    text_features = np.load(os.path.join(cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))
    
    # Validation split is 1
    split_val = 1
    baseline_file = 'val_logits_base.npy'
    baseline_logits_all = np.load(os.path.join(baseline_dir, baseline_file))
    
    # Select sample indices
    split_mask = (split_ids == split_val)
    # Double check no test split (split_id == 2) is used
    assert not np.any(split_ids[split_mask] == 2), "Test split leaked into validation mask!"
    
    split_indices = np.where(split_mask)[0]
    
    if len(split_indices) < args.batch_size:
        raise ValueError(f"Split has only {len(split_indices)} samples, but batch size {args.batch_size} requested.")
        
    batch_indices = split_indices[:args.batch_size]
    # For baseline logits, they are already sliced to validation size, so the relative batch indices are simply 0 to batch_size
    relative_batch_indices = np.arange(args.batch_size)
    
    # Convert batch data to torch tensors
    bx_text = torch.tensor(text_features[batch_indices], dtype=torch.float32).to(device)
    bx_img_g = torch.tensor(image_features_global[batch_indices], dtype=torch.float32).to(device)
    bx_img_p = torch.tensor(image_features_patch[batch_indices], dtype=torch.float32).to(device)
    bx_kg = torch.tensor(kg_features[batch_indices], dtype=torch.float32).to(device)
    bx_rel = torch.tensor(relation_ids[batch_indices], dtype=torch.long).to(device)
    bx_lbl = torch.tensor(labels_fine[batch_indices], dtype=torch.long).to(device)
    bx_logits = torch.tensor(baseline_logits_all[relative_batch_indices], dtype=torch.float32).to(device)
    
    # 3. Instantiate and load frozen F4 Backbone
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_features.shape[1]
    
    f4_model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    f4_ckpt_path = os.path.join(args.project_root, config_data["f4_checkpoint"])
    if os.path.exists(f4_ckpt_path):
        print(f"[+] Loading F4 Backbone: {f4_ckpt_path}")
        f4_ckpt = torch.load(f4_ckpt_path, map_location=device, weights_only=False)
        f4_state = f4_ckpt.get('model_state_dict', f4_ckpt)
        f4_model.load_state_dict(f4_state)
    else:
        raise FileNotFoundError(f"F4 checkpoint not found at {f4_ckpt_path}")
        
    # Instantiate the adapter model wrapping f4_model
    model = StageIE1BottleneckAdapterModel(
        f4_model=f4_model,
        beta=config_data["beta"],
        class_mask=config_data["class_mask"],
        hidden_dim=config_data.get("hidden_dim", 128),
        dropout=config_data.get("dropout", 0.1)
    ).to(device)
    
    # 4. Forward Pass (No grad, no optimization)
    model.eval()
    with torch.no_grad():
        outputs = model(
            text_features=bx_text,
            image_global_features=bx_img_g,
            image_patch_features=bx_img_p,
            kg_features=bx_kg,
            relation_ids=bx_rel,
            baseline_logits=bx_logits
        )
        
    logits_final = outputs['logits_final']
    logits_delta = outputs['logits_delta']
    f4_logits = outputs['f4_logits']
    z_v = outputs['z_v']
    tvcs_score = outputs['tvcs_score']
    
    # 5. Verify tensor shapes
    print("\n[+] Verifying output tensor shapes:")
    print(f"    logits_final: {list(logits_final.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    logits_delta: {list(logits_delta.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    f4_logits:    {list(f4_logits.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    z_v shape:    {list(z_v.shape)} (Expected: [{args.batch_size}, 512])")
    print(f"    tvcs_score:   {list(tvcs_score.shape)} (Expected: [{args.batch_size}])")
    
    assert list(logits_final.shape) == [args.batch_size, 6], "logits_final shape mismatch"
    assert list(logits_delta.shape) == [args.batch_size, 6], "logits_delta shape mismatch"
    assert list(f4_logits.shape) == [args.batch_size, 6], "f4_logits shape mismatch"
    assert list(z_v.shape) == [args.batch_size, 512], "z_v shape mismatch"
    assert list(tvcs_score.shape) == [args.batch_size], "tvcs_score shape mismatch"
    
    # 6. Compute loss components
    # Calculate priors and weights from entire train split
    train_mask = (split_ids == 0)
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_labels) / (6.0 * class_counts)
    
    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    loss_outputs = compute_adapter_total_loss(
        logits_final=logits_final,
        logits_delta=logits_delta,
        f4_logits=f4_logits,
        targets=bx_lbl,
        class_priors=class_priors_t,
        class_weights=class_weights_t,
        bottleneck_classes=config_data["bottleneck_classes"],
        tau_logit_adjust=config_data["tau_logit_adjust"],
        focal_gamma=config_data["focal_gamma"],
        kl_temperature=config_data["kl_temperature"],
        w_balanced=config_data["w_balanced"],
        w_focal=config_data["w_focal"],
        w_ck_guard=config_data["w_ck_guard"],
        w_kl=config_data["w_kl"],
        w_reg=config_data["w_reg"]
    )
    
    loss_total = loss_outputs['loss_total']
    
    # 7. Check NaN/Inf and Sanity
    nan_count, inf_count = 0, 0
    tensors_to_check = {
        "logits_final": logits_final,
        "logits_delta": logits_delta,
        "f4_logits": f4_logits,
        "z_v": z_v,
        "tvcs_score": tvcs_score,
        "loss_total": loss_total
    }
    for name, tensor in tensors_to_check.items():
        nc, ic = check_tensor_sanity(tensor, name)
        nan_count += nc
        inf_count += ic
        if nc > 0 or ic > 0:
            print(f"    [-] WARNING: {name} contains {nc} NaNs and {ic} Infs!")
            
    loss_is_finite = bool(torch.isfinite(loss_total).item())
    print(f"\n[+] Loss is finite: {loss_is_finite}")
    print(f"    Total Loss: {loss_total.item():.6f}")
    print(f"    6way CE Loss: {loss_outputs['loss_6way'].item():.6f}")
    print(f"    Focal Loss: {loss_outputs['loss_focal'].item():.6f}")
    print(f"    CK Guard Loss: {loss_outputs['loss_ck_guard'].item():.6f}")
    print(f"    KL divergence Loss: {loss_outputs['loss_kl'].item():.6f}")
    print(f"    Residual L2 Loss: {loss_outputs['loss_residual'].item():.6f}")
    print(f"    NaN count:  {nan_count}")
    print(f"    Inf count:  {inf_count}")
    
    status = "PASSED" if (loss_is_finite and nan_count == 0 and inf_count == 0) else "FAILED"
    print(f"[+] Dry run status: {status}")
    
    # 8. Save report row in CSV
    out_dryrun_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e", "e1_dryrun")
    os.makedirs(out_dryrun_dir, exist_ok=True)
    report_csv = os.path.join(out_dryrun_dir, "E1_DRYRUN_REPORT.csv")
    
    row_data = {
        "config_name": config_name,
        "beta": config_data["beta"],
        "batch_source": args.split,
        "batch_size": args.batch_size,
        "logits_final_shape": str(list(logits_final.shape)),
        "logits_delta_shape": str(list(logits_delta.shape)),
        "z_v_shape": str(list(z_v.shape)),
        "loss_is_finite": loss_is_finite,
        "loss_total": loss_total.item(),
        "loss_6way": loss_outputs['loss_6way'].item(),
        "loss_focal": loss_outputs['loss_focal'].item(),
        "loss_ck_guard": loss_outputs['loss_ck_guard'].item(),
        "loss_kl": loss_outputs['loss_kl'].item(),
        "loss_residual": loss_outputs['loss_residual'].item(),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "status": status
    }
    
    if os.path.exists(report_csv):
        df = pd.read_csv(report_csv)
        # Drop existing row for the same config name to avoid duplicates
        df = df[df["config_name"] != config_name]
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])
        
    df.to_csv(report_csv, index=False)
    print(f"[+] Saved dry-run result row to: {report_csv}")
    
    # Update dry-run status in E1_PREPARE_SUMMARY.txt
    summary_path = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_e", "E1_PREPARE_SUMMARY.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if "Pending execution of dry-run script" in line:
                continue
            elif f"Dry-run successfully executed for {config_name}:" in line:
                continue
            else:
                new_lines.append(line)
        # Remove trailing empty lines
        while len(new_lines) > 0 and new_lines[-1].strip() == "":
            new_lines.pop()
        # Append the status for this config
        new_lines.append(f"   - Dry-run successfully executed for {config_name}: {status}\n")
        with open(summary_path, 'w') as f:
            f.writelines(new_lines)
        print("[+] Updated E1_PREPARE_SUMMARY.txt with dry-run status.")

if __name__ == "__main__":
    main()
