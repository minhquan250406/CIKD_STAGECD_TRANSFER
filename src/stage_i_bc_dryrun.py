"""
Stage I-BC: Dry-Run Verification script.
Executes a single forward pass under no_grad to verify shapes and loss computation,
runs NaN/Inf checks, and writes results to I_BC_DRYRUN_REPORT.csv.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

from stage_i_bc_multitask_balanced_model import StageIBCMultitaskCIKDPP
from stage_i_bc_losses import compute_multitask_total_loss, check_tensor_sanity

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-BC Multitask Dry-Run Verification.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON config file.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Split to draw batch from (train or val).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for dry-run forward pass.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[+] Running dry-run for config: {args.config} on split: {args.split}")
    
    # 1. Load config file
    if not os.path.exists(args.config):
        print(f"[-] Config file not found: {args.config}")
        sys.exit(1)
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    config_name = config_data.get("config_name", os.path.basename(args.config))
    
    # 2. Check cache and load baseline logits
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    baseline_dir = os.path.join(args.project_root, "outputs", "stage_f0_baseline_anchor")
    
    split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
    relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
    kg_features = np.load(os.path.join(cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
    y_ck = np.load(os.path.join(cache_dir, 'y_ck.npy'))
    text_features = np.load(os.path.join(cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))
    
    if args.split == 'train':
        split_val = 0
        baseline_file = 'train_logits_base.npy'
    else:
        split_val = 1
        baseline_file = 'val_logits_base.npy'
        
    baseline_logits_all = np.load(os.path.join(baseline_dir, baseline_file))
    
    # Select sample indices
    split_mask = (split_ids == split_val)
    split_indices = np.where(split_mask)[0]
    
    if len(split_indices) < args.batch_size:
        raise ValueError(f"Split has only {len(split_indices)} samples, but batch size {args.batch_size} requested.")
        
    batch_indices = split_indices[:args.batch_size]
    relative_batch_indices = np.arange(args.batch_size)
    
    # Convert batch data to torch tensors
    bx_text = torch.tensor(text_features[batch_indices], dtype=torch.float32).to(device)
    bx_img_g = torch.tensor(image_features_global[batch_indices], dtype=torch.float32).to(device)
    bx_img_p = torch.tensor(image_features_patch[batch_indices], dtype=torch.float32).to(device)
    bx_kg = torch.tensor(kg_features[batch_indices], dtype=torch.float32).to(device)
    bx_rel = torch.tensor(relation_ids[batch_indices], dtype=torch.long).to(device)
    bx_lbl = torch.tensor(labels_fine[batch_indices], dtype=torch.long).to(device)
    bx_y_ck = torch.tensor(y_ck[batch_indices], dtype=torch.float32).to(device)
    bx_logits = torch.tensor(baseline_logits_all[relative_batch_indices], dtype=torch.float32).to(device)
    
    # 3. Instantiate model
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_features.shape[1]
    
    model = StageIBCMultitaskCIKDPP(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=config_data.get('dropout', 0.2),
        alpha_init=0.2,
        alpha_max=config_data['alpha_max']
    ).to(device)
    
    # Load checkpoints
    tvcs_ckpt_path = os.path.join(args.project_root, "checkpoints", "stage_f", "tvcs_specialist_seed42_padded_for_f2.pt")
    if os.path.exists(tvcs_ckpt_path):
        print(f"[+] Loading TVCS Specialist: {tvcs_ckpt_path}")
        tvcs_ckpt = torch.load(tvcs_ckpt_path, map_location=device, weights_only=False)
        tvcs_state = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
        model.tvcs_specialist.load_state_dict(tvcs_state)
    else:
        print(f"[-] WARNING: TVCS checkpoint not found at {tvcs_ckpt_path}")
        
    f4_ckpt_path = os.path.join(args.project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
    if os.path.exists(f4_ckpt_path):
        print(f"[+] Loading F4 Backbone: {f4_ckpt_path}")
        f4_ckpt = torch.load(f4_ckpt_path, map_location=device, weights_only=False)
        f4_state = f4_ckpt.get('model_state_dict', f4_ckpt)
        clean_state = {}
        for k, v in f4_state.items():
            if k.startswith('base_model.'):
                clean_state[k.replace('base_model.', '')] = v
            else:
                clean_state[k] = v
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        print(f"    Loaded weights with strict=False. Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        print(f"[-] WARNING: F4 checkpoint not found at {f4_ckpt_path}")
        
    # 4. Forward Pass (No grad, no optimization)
    model.eval()
    with torch.no_grad():
        outputs = model(
            text_features=bx_text,
            image_global_features=bx_img_g,
            image_patch_features=bx_img_p,
            kg_features=bx_kg,
            relation_ids=bx_rel,
            baseline_logits=bx_logits,
            ablation_no_c_emb=True
        )
        
    logits_final = outputs['logits_final']
    logits_delta = outputs['logits_delta']
    binary_logits = outputs['binary_logits']
    ck_real_logits = outputs['ck_real_logits']
    z_v = outputs['z_v']
    
    # 5. Verify tensor shapes
    print("\n[+] Verifying output tensor shapes:")
    print(f"    logits_final:   {list(logits_final.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    logits_delta:   {list(logits_delta.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    binary_logits:  {list(binary_logits.shape)} (Expected: [{args.batch_size}])")
    print(f"    ck_real_logits: {list(ck_real_logits.shape)} (Expected: [{args.batch_size}])")
    print(f"    z_v shape:      {list(z_v.shape)} (Expected: [{args.batch_size}, 512])")
    
    assert list(logits_final.shape) == [args.batch_size, 6], "logits_final shape mismatch"
    assert list(logits_delta.shape) == [args.batch_size, 6], "logits_delta shape mismatch"
    assert list(binary_logits.shape) == [args.batch_size], "binary_logits shape mismatch"
    assert list(ck_real_logits.shape) == [args.batch_size], "ck_real_logits shape mismatch"
    assert list(z_v.shape) == [args.batch_size, 512], "z_v shape mismatch"
    
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
    
    loss_outputs = compute_multitask_total_loss(
        logits_final=logits_final,
        logits_delta=logits_delta,
        binary_logits=binary_logits,
        ck_real_logits=ck_real_logits,
        targets=bx_lbl,
        logits_base=bx_logits,
        class_priors=class_priors_t,
        class_weights=class_weights_t,
        tau_logit_adjust=config_data['tau_logit_adjust'],
        binary_loss_weight=config_data['binary_loss_weight'],
        ck_real_loss_weight=config_data['ck_real_loss_weight'],
        kl_anchor_weight=config_data['kl_anchor_weight'],
        residual_reg_weight=config_data['residual_reg_weight']
    )
    
    loss_total = loss_outputs['loss_total']
    
    # 7. Check NaN/Inf and Sanity
    nan_count, inf_count = 0, 0
    tensors_to_check = {
        "logits_final": logits_final,
        "logits_delta": logits_delta,
        "binary_logits": binary_logits,
        "ck_real_logits": ck_real_logits,
        "z_v": z_v,
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
    print(f"    NaN count:  {nan_count}")
    print(f"    Inf count:  {inf_count}")
    
    status = "PASSED" if (loss_is_finite and nan_count == 0 and inf_count == 0) else "FAILED"
    print(f"[+] Dry run status: {status}")
    
    # 8. Save report row in CSV
    out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement")
    report_csv = os.path.join(out_dir, "I_BC_DRYRUN_REPORT.csv")
    
    row_data = {
        "config_name": config_name,
        "batch_source": args.split,
        "batch_size": args.batch_size,
        "logits_final_shape": str(list(logits_final.shape)),
        "binary_logits_shape": str(list(binary_logits.shape)),
        "ck_real_logits_shape": str(list(ck_real_logits.shape)),
        "z_v_shape": str(list(z_v.shape)),
        "loss_is_finite": loss_is_finite,
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
    
    # Update dry-run status in I_BC_PREPARE_SUMMARY.txt
    summary_path = os.path.join(out_dir, "I_BC_PREPARE_SUMMARY.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if "Pending execution of dry-run script." in line:
                new_lines.append(f"   - Dry-run successfully executed for {config_name}: {status}\n")
            else:
                new_lines.append(line)
        with open(summary_path, 'w') as f:
            f.writelines(new_lines)
        print("[+] Updated I_BC_PREPARE_SUMMARY.txt with dry-run status.")

if __name__ == "__main__":
    main()
