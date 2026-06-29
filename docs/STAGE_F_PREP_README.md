# Stage F: CIKD++ Residual TVCS-Transformer (CIKD++-RT) Preparation

This document outlines the preparation steps, architecture, dry-run commands, future execution instructions, and safety guardrails for the **CIKD++ Residual TVCS-Transformer (CIKD++-RT)** model.

## 1. Architecture Overview

CIKD++-RT uses the existing `text_image_kg_concat` baseline as a frozen anchor model to produce baseline logits, and trains a new residual correction branch.

### Mathematical Formula
$$logits_{final} = logits_{base} + \alpha \times logits_{\delta}$$

Where:
* $logits_{base}$ are the predictions from the frozen `text_image_kg_concat` baseline anchor.
* $\alpha$ is a learnable scalar parameter scaled via sigmoid and bounded by $\alpha_{max}$ (default: 0.5):
  $$\alpha = \sigma(\alpha_{raw}) \times \alpha_{max}$$
* $logits_{\delta}$ is the output of the **Residual Transformer Fusion** module.

### Core Modules
1. **TVCS Specialist (`TVCSSpecialist`)**: Projects KG + relation into query space ($Q$), image patches into key ($K$) and value ($V$) spaces. Applies dot-product attention over patch tokens to extract visual evidence $z_v$ (dim: 512). Outputs contradiction logit $c\_logit$ and embedding $c\_emb$ (dim: 64) via sub-MLPs.
2. **Residual Transformer Fusion (`ResidualTransformerFusion`)**: Projects 7 inputs into $d\_model$ (default: 256) token embeddings:
   - Text features ($[B, 768]$)
   - Global image features ($[B, 512]$)
   - KG features ($[B, 100]$)
   - Relation embedding ($[B, 64]$)
   - TVCS visual evidence $z_v$ ($[B, 512]$)
   - Contradiction embedding $c\_emb$ ($[B, 64]$)
   - Baseline logits ($[B, 6]$)
   Fuses these 7 tokens via a 2-layer `TransformerEncoder` (4 attention heads) and mean-pools them to decode residual logits $logits_{\delta}$.

---

## 2. Dry-Run Verification Commands
These commands perform full safety/sanity checks (file presence, array shape validation, checkpoint CPU inspection, and forward smoke tests) without running training or evaluations.

### F0: Export Baseline Logits Dry-Run
```bash
python src/stage_f0_export_baseline_logits.py ^
  --cache_dir data/cache/kg_complete ^
  --checkpoint checkpoints/baselines/text_image_kg_concat_seed42.pt ^
  --out_dir outputs/stage_f0_baseline_anchor ^
  --batch_size 256 ^
  --seed 42 ^
  --dry_run
```

### F1: Train TVCS Specialist Dry-Run
```bash
python src/stage_f1_train_tvcs_specialist.py ^
  --cache_dir data/cache/tvcs_eligible ^
  --out_dir outputs/stage_f1_tvcs_specialist ^
  --checkpoint_out checkpoints/stage_f/tvcs_specialist_seed42.pt ^
  --epochs 20 ^
  --batch_size 128 ^
  --lr 1e-4 ^
  --weight_decay 1e-4 ^
  --patience 5 ^
  --seed 42 ^
  --dry_run
```

### F2: Train Residual Transformer Dry-Run
```bash
python src/stage_f2_train_residual_transformer.py ^
  --cache_dir data/cache/kg_complete ^
  --baseline_logits_dir outputs/stage_f0_baseline_anchor ^
  --tvcs_checkpoint checkpoints/stage_f/tvcs_specialist_seed42.pt ^
  --out_dir outputs/stage_f2_cikd_pp_rt ^
  --checkpoint_out checkpoints/stage_f/cikd_pp_rt_seed42.pt ^
  --epochs 20 ^
  --batch_size 128 ^
  --lr 1e-4 ^
  --weight_decay 1e-4 ^
  --d_model 256 ^
  --num_layers 2 ^
  --num_heads 4 ^
  --dropout 0.2 ^
  --alpha_init 0.2 ^
  --alpha_max 0.5 ^
  --lambda_tvcs 0.5 ^
  --focal_gamma 1.0 ^
  --residual_mu 0.01 ^
  --patience 5 ^
  --seed 42 ^
  --dry_run
```

### F3: Ablation Study Runner Dry-Run
```bash
python src/stage_f3_ablation.py ^
  --cache_dir data/cache/kg_complete ^
  --baseline_logits_dir outputs/stage_f0_baseline_anchor ^
  --tvcs_checkpoint checkpoints/stage_f/tvcs_specialist_seed42.pt ^
  --out_dir outputs/stage_f3_ablation ^
  --configs full,no_tvcs_loss,no_c_emb,no_residual,global_only ^
  --seed 42 ^
  --dry_run
```

### F4: Final Lock Evaluation Dry-Run
```bash
python src/stage_f4_final_lock.py ^
  --cache_dir data/cache/kg_complete ^
  --baseline_logits_dir outputs/stage_f0_baseline_anchor ^
  --checkpoint checkpoints/stage_f/cikd_pp_rt_best.pt ^
  --out_dir outputs/stage_f4_final_lock ^
  --bootstrap 1000 ^
  --seed 42 ^
  --dry_run
```

---

## 3. Future Execution Commands (Training and Evaluation)
To run actual model training and testing, add `--execute` / `--execute_train` / `--execute_final` to the respective commands:

### F0: Export Baseline Logits
```bash
python src/stage_f0_export_baseline_logits.py --cache_dir data/cache/kg_complete --checkpoint checkpoints/baselines/text_image_kg_concat_seed42.pt --out_dir outputs/stage_f0_baseline_anchor --execute
```

### F1: Train TVCS Specialist
```bash
python src/stage_f1_train_tvcs_specialist.py --cache_dir data/cache/tvcs_eligible --out_dir outputs/stage_f1_tvcs_specialist --checkpoint_out checkpoints/stage_f/tvcs_specialist_seed42.pt --execute_train
```

### F2: Train Residual Transformer (CIKD++-RT)
```bash
python src/stage_f2_train_residual_transformer.py --cache_dir data/cache/kg_complete --baseline_logits_dir outputs/stage_f0_baseline_anchor --tvcs_checkpoint checkpoints/stage_f/tvcs_specialist_seed42.pt --out_dir outputs/stage_f2_cikd_pp_rt --checkpoint_out checkpoints/stage_f/cikd_pp_rt_seed42.pt --execute_train
```

### F3: Run Ablation Study
```bash
python src/stage_f3_ablation.py --cache_dir data/cache/kg_complete --baseline_logits_dir outputs/stage_f0_baseline_anchor --tvcs_checkpoint checkpoints/stage_f/tvcs_specialist_seed42.pt --out_dir outputs/stage_f3_ablation --configs full,no_tvcs_loss,no_c_emb,no_residual,global_only --execute_train
```

### F4: Final Lock evaluation
```bash
python src/stage_f4_final_lock.py --cache_dir data/cache/kg_complete --baseline_logits_dir outputs/stage_f0_baseline_anchor --checkpoint checkpoints/stage_f/cikd_pp_rt_seed42.pt --out_dir outputs/stage_f4_final_lock --bootstrap 1000 --execute_final
```

---

## 4. Expected Outputs

### Stage F0
* `outputs/stage_f0_baseline_anchor/train_logits_base.npy`
* `outputs/stage_f0_baseline_anchor/val_logits_base.npy`
* `outputs/stage_f0_baseline_anchor/test_logits_base.npy`
* `outputs/stage_f0_baseline_anchor/f0_baseline_anchor_metrics.csv`

### Stage F1
* `checkpoints/stage_f/tvcs_specialist_seed42.pt`
* `outputs/stage_f1_tvcs_specialist/f1_tvcs_metrics_val.csv`
* `outputs/stage_f1_tvcs_specialist/f1_tvcs_scores_val.csv`
* `outputs/stage_f1_tvcs_specialist/f1_tvcs_hist_val.png`
* `outputs/stage_f1_tvcs_specialist/f1_tvcs_summary.txt`

### Stage F2
* `checkpoints/stage_f/cikd_pp_rt_seed42.pt`
* `outputs/stage_f2_cikd_pp_rt/f2_rt_metrics_val.csv`
* `outputs/stage_f2_cikd_pp_rt/f2_rt_summary.txt`

### Stage F3
* `outputs/stage_f3_ablation/full/metrics_full.csv`
* `outputs/stage_f3_ablation/no_tvcs_loss/metrics_no_tvcs_loss.csv`
* `outputs/stage_f3_ablation/no_c_emb/metrics_no_c_emb.csv`
* `outputs/stage_f3_ablation/no_residual/metrics_no_residual.csv`
* `outputs/stage_f3_ablation/global_only/metrics_global_only.csv`

### Stage F4
* `outputs/stage_f4_final_lock/04_final_main_metrics.csv`
* `outputs/stage_f4_final_lock/04_final_per_class_f1.csv`
* `outputs/stage_f4_final_lock/04_final_tvcs_metrics.csv`
* `outputs/stage_f4_final_lock/04_bootstrap_ci_vs_tikg.csv`
* `outputs/stage_f4_final_lock/04_bootstrap_ci_vs_old_cikd.csv`
* `outputs/stage_f4_final_lock/04_confusion_matrix.png`
* `outputs/stage_f4_final_lock/04_tvcs_hist_test.png`
* `outputs/stage_f4_final_lock/04_STAGE_F_FINAL_LOCK_SUMMARY.txt`

---

## 5. Safety Guardrails Summary
* **No Automatic execution**: Training runs and lock evaluation are blocked unless explicit `--execute_train` / `--execute_final` / `--execute` flags are provided.
* **No Overwrites**: Unless `--overwrite` is explicitly appended, all scripts refuse to overwrite any existing output files and exit cleanly.
* **Test Isolation**: Validation selection score computation is strictly isolated to the validation split (`split_ids == 1`). The test split (`split_ids == 2`) is exclusively loaded during final lock evaluation (Stage F4).
* **Stage A-E Protection**: No files in `outputs/stage_e_final_lock` or other early stage paths are touched, modified, or overwritten by any Stage F scripts.
