"""
Stage I-F2: Safe Refinement Dry-Run.
Loads validation batch only, performs shape verification and loss sanity checks,
and ensures no gradients, training steps, or test set evaluations occur.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

# Add workspace path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cikd_pp_rt import CIKDPPResidualTransformer
from src.stage_i_f1_feature_refresh_model import StageIFFeatureRefreshCIKDPP
from src.stage_i_f1_losses import compute_total_loss, check_tensor_sanity

def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run Stage I-F2 configurations.")
    parser.add_argument("--project_root", type=str, default="D:\\CIKD_STAGECD_TRANSFER",
                        help="Root directory of the project.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "validation"],
                        help="Split to use (must be val/validation).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for the dry-run.")
    parser.add_argument("--no_test_eval", action="store_true", required=True,
                        help="Enforce safety gate: must be present to ensure test set isolation.")
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.no_test_eval, "Abort: --no_test_eval flag is missing! Must be present to ensure test set isolation."
    
    # Strict validation check
    if args.split not in ["val", "validation"]:
        print(f"ERROR: Split '{args.split}' is not allowed. Only validation split is permitted.")
        sys.exit(1)
        
    print("========================================================================")
    print("STAGE I-F2: SAFE REFINEMENT DRY-RUN")
    print("========================================================================")
    print(f"Project root: {args.project_root}")
    print(f"Split:        {args.split}")
    print(f"Batch size:   {args.batch_size}")
    print("------------------------------------------------------------------------")
    print("CRITICAL CHECK: GRADIENTS, OPTIMIZER, AND TEST SET ENTIRELY ISOLATED")
    print("------------------------------------------------------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device selected: {device}")

    # 1. Load dataset cache
    cache_dir = os.path.join(args.project_root, "data", "cache", "kg_complete")
    print(f"[+] Loading dataset cache from: {cache_dir}")
    
    split_ids = np.load(os.path.join(cache_dir, 'split_ids.npy'))
    
    # Enforce no split_id == 2 (test set) is ever loaded or used
    if 2 in split_ids[split_ids == 2]:
        print("[!] Warning: test set split_id == 2 exists in cache. Isolating test set...")
        
    val_mask = (split_ids == 1)
    if np.sum(val_mask) == 0:
        print("ERROR: Validation split (split_id == 1) has 0 samples in cache.")
        sys.exit(1)
        
    # Double check no split_id == 2 gets in
    assert not np.any(split_ids[val_mask] == 2), "Test split leaked into validation split!"

    relation_ids = np.load(os.path.join(cache_dir, 'relation_ids.npy'))
    kg_features = np.load(os.path.join(cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(cache_dir, 'labels_fine.npy'))
    text_features = np.load(os.path.join(cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(cache_dir, 'image_features_patch.npy'))
    
    # Load baseline logits from outputs/stage_f0_baseline_anchor
    baseline_dir = os.path.join(args.project_root, "outputs", "stage_f0_baseline_anchor")
    print(f"[+] Loading baseline anchor logits from: {baseline_dir}")
    val_logits_base = np.load(os.path.join(baseline_dir, 'val_logits_base.npy'))
    
    # Get a single batch from the validation split
    B = args.batch_size
    bx_text = torch.tensor(text_features[val_mask][:B], dtype=torch.float32).to(device)
    bx_img_g = torch.tensor(image_features_global[val_mask][:B], dtype=torch.float32).to(device)
    bx_img_p = torch.tensor(image_features_patch[val_mask][:B], dtype=torch.float32).to(device)
    bx_kg = torch.tensor(kg_features[val_mask][:B], dtype=torch.float32).to(device)
    bx_rel = torch.tensor(relation_ids[val_mask][:B], dtype=torch.long).to(device)
    bx_lbl = torch.tensor(labels_fine[val_mask][:B], dtype=torch.long).to(device)
    bx_logits = torch.tensor(val_logits_base[:B], dtype=torch.float32).to(device)

    print(f"[+] Loaded validation batch of size {B}.")
    
    # 2. Compute class priors and weights from train split (split_id == 0) for loss sanity check
    train_mask = (split_ids == 0)
    train_lbl_np = labels_fine[train_mask]
    class_counts = np.bincount(train_lbl_np, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    class_priors = class_counts / class_counts.sum()
    class_weights = len(train_lbl_np) / (6.0 * class_counts)
    
    class_priors_t = torch.tensor(class_priors, dtype=torch.float32).to(device)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 3. Locate configurations under configs/stage_i_f/f2_refinement/
    f2_config_dir = os.path.join(args.project_root, "configs", "stage_i_f", "f2_refinement")
    config_names = [
        "i_f2_a_gamma008_lr5e5_kl015_l205",
        "i_f2_b_gamma010_lr5e5_kl020_l208",
        "i_f2_c_gamma012_lr5e5_kl015_l205",
        "i_f2_d_gamma010_lr1e4_kl020_l208",
        "i_f2_e_gamma005_lr5e5_kl020_l210"
    ]

    report_rows = []
    
    # 4. Instantiate shared backbone models
    num_relations = int(relation_ids.max()) + 1
    kg_dim = kg_features.shape[1]
    
    print("[+] Initializing backbone models...")
    # F4 backbone
    f4_model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    # We load backbone checkpoint from default paths
    f4_ckpt_path = os.path.join(args.project_root, "outputs", "stage_f3_ablation", "no_c_emb", "cikd_pp_rt_ablation_no_c_emb.pt")
    if os.path.exists(f4_ckpt_path):
        print(f"    - Loading F4 Backbone: {f4_ckpt_path}")
        f4_ckpt = torch.load(f4_ckpt_path, map_location=device, weights_only=False)
        f4_state = f4_ckpt.get('model_state_dict', f4_ckpt)
        f4_model.load_state_dict(f4_state)
    else:
        raise FileNotFoundError(f"F4 Backbone checkpoint not found at {f4_ckpt_path}")
        
    tvcs_ckpt_path = os.path.join(args.project_root, "checkpoints", "stage_f", "tvcs_specialist_seed42_padded_for_f2.pt")
    if os.path.exists(tvcs_ckpt_path):
        print(f"    - Loading TVCS Specialist checkpoint: {tvcs_ckpt_path}")
        tvcs_ckpt = torch.load(tvcs_ckpt_path, map_location=device, weights_only=False)
        tvcs_state = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
        f4_model.tvcs_specialist.load_state_dict(tvcs_state)
    else:
        raise FileNotFoundError(f"TVCS checkpoint not found at {tvcs_ckpt_path}")
        
    # Set backbones to eval mode
    f4_model.eval()
    for param in f4_model.parameters():
        param.requires_grad = False

    print("\n[+] Running dry-run sweep across Stage I-F2 configurations...")
    
    for name in config_names:
        cfg_path = os.path.join(f2_config_dir, f"{name}.json")
        assert os.path.exists(cfg_path), f"Configuration file missing: {cfg_path}"
        
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
            
        print(f"\n  Configuration: {name}")
        print("  " + "-" * (len(name) + 15))
        
        # Instantiate Feature Refresh model
        model = StageIFFeatureRefreshCIKDPP(
            f4_model=f4_model,
            num_relations=num_relations,
            kg_dim=kg_dim,
            relation_emb_dim=64,
            gamma=cfg["gamma"],
            use_patch_adapter=cfg.get("use_patch_adapter", True),
            use_kg_relation_adapter=cfg.get("use_kg_relation_adapter", True),
            use_tvcs_zv=cfg.get("use_tvcs_zv", True),
            d_model=cfg.get("d_model", 256),
            num_layers=cfg.get("num_layers", 2),
            num_heads=cfg.get("num_heads", 4),
            dropout=cfg.get("dropout", 0.1)
        ).to(device)
        
        model.eval()
        
        # Run forward pass (NO GRADIENTS)
        with torch.no_grad():
            outputs = model(
                text_features=bx_text,
                image_global_features=bx_img_g,
                image_patch_features=bx_img_p,
                kg_features=bx_kg,
                relation_ids=bx_rel,
                baseline_logits=bx_logits
            )
            
            # Verify shapes
            f4_logits_shape = list(outputs['f4_logits'].shape)
            delta_new_shape = list(outputs['delta_new'].shape)
            final_logits_shape = list(outputs['logits_final'].shape)
            text_refresh_shape = list(outputs['text_refresh'].shape)
            
            patch_refresh_shape = list(outputs['patch_refresh'].shape) if outputs.get('patch_refresh') is not None else [0]
            kg_refresh_shape = list(outputs['kg_refresh'].shape) if outputs.get('kg_refresh') is not None else [0]
            
            z_v_shape = list(outputs['z_v'].shape)
            tvcs_score_shape = list(outputs['tvcs_score'].shape)
            
            # Expected shapes assertion
            assert f4_logits_shape == [B, 6], f"f4_logits shape mismatch: {f4_logits_shape}"
            assert delta_new_shape == [B, 6], f"delta_new shape mismatch: {delta_new_shape}"
            assert final_logits_shape == [B, 6], f"final_logits shape mismatch: {final_logits_shape}"
            assert text_refresh_shape == [B, 256], f"text_refresh shape mismatch: {text_refresh_shape}"
            assert patch_refresh_shape == [B, 256], f"patch_refresh shape mismatch: {patch_refresh_shape}"
            assert kg_refresh_shape == [B, 256], f"kg_refresh shape mismatch: {kg_refresh_shape}"
            assert z_v_shape == [B, 512], f"z_v shape mismatch: {z_v_shape}"
            assert tvcs_score_shape == [B], f"tvcs_score shape mismatch: {tvcs_score_shape}"
            
            print("    [OK] Shape checks passed.")
            
            # Compute planned loss once for numerical sanity check
            loss_outputs = compute_total_loss(
                logits_final=outputs['logits_final'],
                delta_new=outputs['delta_new'],
                f4_logits=outputs['f4_logits'],
                targets=bx_lbl,
                class_priors=class_priors_t,
                class_weights=class_weights_t,
                bottleneck_classes=cfg.get("loss_focus_classes", [1, 2, 5]),
                tau_logit_adjust=cfg.get("tau_logit_adjust", 0.5),
                focal_gamma=cfg.get("focal_gamma", 1.5),
                kl_temperature=cfg.get("kl_temperature", 1.0),
                w_balanced=cfg.get("w_balanced", 1.00),
                w_focal=cfg.get("w_focal", 0.20),
                w_ck_guard=cfg.get("w_ck_guard", 0.10),
                w_kl=cfg.get("w_kl", 0.15),
                w_delta_norm=cfg.get("w_delta_norm", 0.05)
            )
            
            # Check NaN/Inf in all relevant tensors
            nan_inf_occurred = False
            tensors_to_check = {
                "logits_final": outputs['logits_final'],
                "delta_new": outputs['delta_new'],
                "f4_logits": outputs['f4_logits'],
                "loss_total": loss_outputs['loss_total'],
                "text_refresh": outputs['text_refresh'],
                "patch_refresh": outputs['patch_refresh'],
                "kg_refresh": outputs['kg_refresh'],
                "z_v": outputs['z_v'],
                "tvcs_score": outputs['tvcs_score']
            }
            
            for t_name, tensor in tensors_to_check.items():
                nc, ic = check_tensor_sanity(tensor, t_name)
                if nc > 0 or ic > 0:
                    nan_inf_occurred = True
                    print(f"      [!] Sanity check FAILED for {t_name}: {nc} NaNs, {ic} Infs found!")
            
            if not nan_inf_occurred:
                print("    [OK] Sanity checks (no NaN/Inf) passed.")
                print(f"    [OK] Loss values calculated successfully: Total Loss = {loss_outputs['loss_total'].item():.4f}")
            else:
                print("    [ERROR] NaN/Inf values detected!")
                
            report_rows.append({
                "config_name": name,
                "f4_logits_shape": str(f4_logits_shape),
                "delta_new_shape": str(delta_new_shape),
                "final_logits_shape": str(final_logits_shape),
                "text_refresh_shape": str(text_refresh_shape),
                "patch_refresh_shape": str(patch_refresh_shape),
                "kg_refresh_shape": str(kg_refresh_shape),
                "z_v_shape": str(z_v_shape),
                "tvcs_score_shape": str(tvcs_score_shape),
                "loss_total": float(loss_outputs['loss_total'].item()),
                "loss_6way": float(loss_outputs['loss_6way'].item()),
                "loss_focal": float(loss_outputs['loss_focal'].item()),
                "loss_ck_guard": float(loss_outputs['loss_ck_guard'].item()),
                "loss_kl": float(loss_outputs['loss_kl'].item()),
                "loss_residual": float(loss_outputs['loss_residual'].item()),
                "nan_inf_detected": nan_inf_occurred
            })

    # 5. Save Report CSV
    f2_out_dir = os.path.join(args.project_root, "outputs", "stage_i_macro_micro_improvement", "stage_i_f", "f2_refinement")
    report_csv_path = os.path.join(f2_out_dir, "F2_DRYRUN_REPORT.csv")
    df_report = pd.DataFrame(report_rows)
    df_report.to_csv(report_csv_path, index=False)
    print(f"\n[+] Saved dry-run report to: {report_csv_path}")

    # 6. Save Summary TXT
    summary_txt_path = os.path.join(f2_out_dir, "F2_DRYRUN_SUMMARY.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("========================================================================\n")
        f.write("STAGE I-F2: SAFE REFINEMENT DRY-RUN SUMMARY\n")
        f.write("========================================================================\n\n")
        f.write(f"Project Root: {args.project_root}\n")
        f.write(f"Split:        {args.split}\n")
        f.write(f"Batch size:   {args.batch_size}\n\n")
        
        f.write("CRITICAL RESTRICTION CONFIRMATIONS:\n")
        f.write("  1. Gradients calculated?            NO (torch.no_grad used exclusively)\n")
        f.write("  2. Optimizer steps called?          NO\n")
        f.write("  3. Backward pass performed?         NO\n")
        f.write("  4. Locked test split evaluated?     NO\n")
        f.write("  5. split_id == 2 used?              NO\n\n")
        
        f.write("DETAILED RESULTS CONFIGURATION BY CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(df_report.to_string(index=False))
        f.write("\n" + "-" * 80 + "\n\n")
        f.write("All shape and numerical sanity checks successfully executed.\n")
        
    print(f"[+] Saved dry-run summary to: {summary_txt_path}")
    print("[+] Stage I-F2 Dry-Run Complete.")

if __name__ == "__main__":
    main()
