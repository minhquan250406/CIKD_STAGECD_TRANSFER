"""
Stage I-F1: Dry-Run Verification Script.
Loads a validation batch under no_grad, performs a forward pass,
verifies all required shapes, checks for NaN/Infs, computes planned loss,
and outputs a CSV report and a TXT summary.
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
from src.stage_i_f1_feature_refresh_model import StageIFFeatureRefreshCIKDPP
from src.stage_i_f1_losses import compute_total_loss, check_tensor_sanity

def parse_args():
    parser = argparse.ArgumentParser(description="Stage I-F1 Dry-Run")
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

    # 2. Check cache and load validation data
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

    # Select validation sample indices
    split_mask = (split_ids == split_val)
    # Double check no test split (split_id == 2) is leaked
    assert not np.any(split_ids[split_mask] == 2), "Test split leaked into validation mask!"

    split_indices = np.where(split_mask)[0]
    if len(split_indices) < args.batch_size:
        raise ValueError(f"Split has only {len(split_indices)} samples, but batch size {args.batch_size} requested.")

    batch_indices = split_indices[:args.batch_size]
    # For baseline logits, they are already sliced to validation size, so the relative batch indices are simple:
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

    # 4. Instantiate the Stage I-F Feature Refresh Model
    model = StageIFFeatureRefreshCIKDPP(
        f4_model=f4_model,
        num_relations=num_relations,
        kg_dim=kg_dim,
        relation_emb_dim=64,
        gamma=config_data["gamma"],
        use_patch_adapter=config_data.get("use_patch_adapter", True),
        use_kg_relation_adapter=config_data.get("use_kg_relation_adapter", True),
        use_tvcs_zv=config_data.get("use_tvcs_zv", True),
        d_model=config_data.get("d_model", 256),
        num_layers=config_data.get("num_layers", 2),
        num_heads=config_data.get("num_heads", 4),
        dropout=config_data.get("dropout", 0.1)
    ).to(device)

    # 5. Forward Pass (no_grad, no backward, no optimizer steps)
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
    delta_new = outputs['delta_new']
    f4_logits = outputs['f4_logits']
    text_refresh = outputs['text_refresh']
    patch_refresh = outputs['patch_refresh']
    kg_refresh = outputs['kg_refresh']
    z_v = outputs['z_v']
    tvcs_score = outputs['tvcs_score']

    # 6. Verify shapes and constraints
    print("\n[+] Verifying output tensor shapes:")
    print(f"    logits_final: {list(logits_final.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    delta_new:    {list(delta_new.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    f4_logits:    {list(f4_logits.shape)} (Expected: [{args.batch_size}, 6])")
    print(f"    text_refresh: {list(text_refresh.shape)} (Expected: [{args.batch_size}, 256])")
    if patch_refresh is not None:
        print(f"    patch_refresh:{list(patch_refresh.shape)} (Expected: [{args.batch_size}, 256])")
    else:
        print("    patch_refresh: None (Ablation enabled)")
    if kg_refresh is not None:
        print(f"    kg_refresh:   {list(kg_refresh.shape)} (Expected: [{args.batch_size}, 256])")
    print(f"    z_v:          {list(z_v.shape)} (Expected: [{args.batch_size}, 512])")
    print(f"    tvcs_score:   {list(tvcs_score.shape)} (Expected: [{args.batch_size}])")

    shape_failures = []
    if list(logits_final.shape) != [args.batch_size, 6]:
        shape_failures.append(f"logits_final shape {list(logits_final.shape)} is not {[args.batch_size, 6]}")
    if list(delta_new.shape) != [args.batch_size, 6]:
        shape_failures.append(f"delta_new shape {list(delta_new.shape)} is not {[args.batch_size, 6]}")
    if list(f4_logits.shape) != [args.batch_size, 6]:
        shape_failures.append(f"f4_logits shape {list(f4_logits.shape)} is not {[args.batch_size, 6]}")
    if list(text_refresh.shape) != [args.batch_size, 256]:
        shape_failures.append(f"text_refresh shape {list(text_refresh.shape)} is not {[args.batch_size, 256]}")
    if config_data.get("use_patch_adapter", True) and (patch_refresh is None or list(patch_refresh.shape) != [args.batch_size, 256]):
        shape_failures.append("patch_refresh shape is not valid")
    if config_data.get("use_kg_relation_adapter", True) and (kg_refresh is None or list(kg_refresh.shape) != [args.batch_size, 256]):
        shape_failures.append("kg_refresh shape is not valid")
    if list(z_v.shape) != [args.batch_size, 512]:
        shape_failures.append(f"z_v shape {list(z_v.shape)} is not {[args.batch_size, 512]}")
    if list(tvcs_score.shape) != [args.batch_size]:
        shape_failures.append(f"tvcs_score shape {list(tvcs_score.shape)} is not {[args.batch_size]}")

    # 7. Compute planned loss components
    # Calculate class priors and weights from train split to match project conventions
    train_mask = (split_ids == 0)
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_labels) / (6.0 * class_counts)

    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    loss_outputs = compute_total_loss(
        logits_final=logits_final,
        delta_new=delta_new,
        f4_logits=f4_logits,
        targets=bx_lbl,
        class_priors=class_priors_t,
        class_weights=class_weights_t,
        bottleneck_classes=config_data.get("loss_focus_classes", [1, 2, 5]),
        tau_logit_adjust=config_data.get("tau_logit_adjust", 0.5),
        focal_gamma=config_data.get("focal_gamma", 1.5),
        kl_temperature=config_data.get("kl_temperature", 1.0),
        w_balanced=config_data.get("w_balanced", 1.00),
        w_focal=config_data.get("w_focal", 0.20),
        w_ck_guard=config_data.get("w_ck_guard", 0.10),
        w_kl=config_data.get("w_kl", 0.10),
        w_delta_norm=config_data.get("w_delta_norm", 0.03)
    )

    loss_total = loss_outputs['loss_total']

    # 8. Check NaNs and Infs
    nan_count, inf_count = 0, 0
    tensors_to_check = {
        "logits_final": logits_final,
        "delta_new": delta_new,
        "f4_logits": f4_logits,
        "text_refresh": text_refresh,
        "z_v": z_v,
        "tvcs_score": tvcs_score,
        "loss_total": loss_total
    }
    if patch_refresh is not None:
        tensors_to_check["patch_refresh"] = patch_refresh
    if kg_refresh is not None:
        tensors_to_check["kg_refresh"] = kg_refresh

    for name, tensor in tensors_to_check.items():
        nc, ic = check_tensor_sanity(tensor, name)
        nan_count += nc
        inf_count += ic
        if nc > 0 or ic > 0:
            print(f"    [-] WARNING: {name} contains {nc} NaNs and {ic} Infs!")

    loss_is_finite = bool(torch.isfinite(loss_total).item())
    print(f"\n[+] Loss verification:")
    print(f"    Total Loss: {loss_total.item():.6f}")
    print(f"    6way Balanced CE: {loss_outputs['loss_6way'].item():.6f}")
    print(f"    Focal Loss (1,2,5): {loss_outputs['loss_focal'].item():.6f}")
    print(f"    CK Guard Loss: {loss_outputs['loss_ck_guard'].item():.6f}")
    print(f"    KL to F4 Loss: {loss_outputs['loss_kl'].item():.6f}")
    print(f"    Delta Norm Loss: {loss_outputs['loss_residual'].item():.6f}")
    print(f"    Loss is finite: {loss_is_finite}")

    status = "PASSED" if (loss_is_finite and nan_count == 0 and inf_count == 0 and len(shape_failures) == 0) else "FAILED"
    print(f"[+] Dry run status: {status}")

    # 9. Save report row in CSV
    out_dryrun_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f", "f1_dryrun")
    os.makedirs(out_dryrun_dir, exist_ok=True)
    report_csv = os.path.join(out_dryrun_dir, "F1_DRYRUN_REPORT.csv")

    row_data = {
        "config_name": config_name,
        "gamma": config_data["gamma"],
        "batch_source": args.split,
        "batch_size": args.batch_size,
        "logits_final_shape": str(list(logits_final.shape)),
        "delta_new_shape": str(list(delta_new.shape)),
        "z_v_shape": str(list(z_v.shape)),
        "loss_is_finite": loss_is_finite,
        "loss_total": loss_total.item(),
        "loss_6way": loss_outputs['loss_6way'].item(),
        "loss_focal": loss_outputs['loss_focal'].item(),
        "loss_ck_guard": loss_outputs['loss_ck_guard'].item(),
        "loss_kl": loss_outputs['loss_kl'].item(),
        "loss_delta_norm": loss_outputs['loss_residual'].item(),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "status": status
    }

    if os.path.exists(report_csv):
        df = pd.read_csv(report_csv)
        df = df[df["config_name"] != config_name]
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])

    df.to_csv(report_csv, index=False)
    print(f"[+] Saved dry-run result row to: {report_csv}")

    # 10. Write dry-run summary text file
    summary_path = os.path.join(out_dryrun_dir, "F1_DRYRUN_SUMMARY.txt")
    
    # Read existing summary lines if they exist, to list statuses of all dry-runned configs
    existing_runs = {}
    if os.path.exists(summary_path):
        try:
            # We can parse the CSV report to summarize all executed configs!
            df_all = pd.read_csv(report_csv)
            for _, r in df_all.iterrows():
                existing_runs[r["config_name"]] = r["status"]
        except Exception:
            pass
    existing_runs[config_name] = status

    with open(summary_path, "w") as f_sum:
        f_sum.write("========================================================================\n")
        f_sum.write("STAGE I-F1 DRY-RUN SUMMARY REPORT\n")
        f_sum.write("========================================================================\n\n")
        f_sum.write("CRITICAL ASSURANCES:\n")
        f_sum.write("--------------------\n")
        f_sum.write("- NO TRAINING WAS RUN (no backpropagation, no optimizer step).\n")
        f_sum.write("- LOCKED TEST WAS NOT EVALUATED (validation split ONLY used).\n\n")
        
        f_sum.write("SHAPE VALIDATION AND SANITY CHECK:\n")
        f_sum.write("----------------------------------\n")
        if len(shape_failures) == 0:
            f_sum.write("- All forward pass shapes are VALID.\n")
        else:
            f_sum.write("- SHAPE FAILURES DETECTED:\n")
            for fail in shape_failures:
                f_sum.write(f"  * {fail}\n")
        
        f_sum.write(f"- NaN count: {nan_count}\n")
        f_sum.write(f"- Inf count: {inf_count}\n\n")
        
        f_sum.write("DRY-RUN STATUS PER CONFIG:\n")
        f_sum.write("--------------------------\n")
        for name, run_status in existing_runs.items():
            f_sum.write(f"- Config '{name}': {run_status}\n")
            
        f_sum.write("\nREADINESS FOR VALIDATION-ONLY TRAINING:\n")
        f_sum.write("----------------------------------------\n")
        if all(s == "PASSED" for s in existing_runs.values()):
            f_sum.write("- YES, Stage I-F is ready for validation-only training later.\n")
        else:
            f_sum.write("- NO, there are failed configurations. Please address issues before proceeding.\n")

    print(f"[+] Saved dry-run summary to: {summary_path}")

if __name__ == "__main__":
    main()
