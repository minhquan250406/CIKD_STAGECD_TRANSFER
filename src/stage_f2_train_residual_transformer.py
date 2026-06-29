"""
Stage F2: Train CIKD++ Residual TVCS-Transformer.
Prepares code to train the CIKD++-RT model on the kg_complete cached subset.
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import model
from models.cikd_pp_rt import CIKDPPResidualTransformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train CIKD++ Residual TVCS-Transformer")
    parser.add_argument('--cache_dir', type=str, default='data/cache/kg_complete', help='Cache directory')
    parser.add_argument('--baseline_logits_dir', type=str, default='outputs/stage_f0_baseline_anchor', help='Baseline logits directory')
    parser.add_argument('--tvcs_checkpoint', type=str, default='checkpoints/stage_f/tvcs_specialist_seed42.pt', help='TVCS specialist checkpoint')
    parser.add_argument('--out_dir', type=str, default='outputs/stage_f2_cikd_pp_rt', help='Output directory')
    parser.add_argument('--checkpoint_out', type=str, default='checkpoints/stage_f/cikd_pp_rt_seed42.pt', help='Checkpoint output path')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer num layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Transformer num heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
    parser.add_argument('--alpha_init', type=float, default=0.2, help='Residual scalar alpha initial value')
    parser.add_argument('--alpha_max', type=float, default=0.5, help='Residual scalar alpha max bound')
    
    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lambda_tvcs', type=float, default=0.5, help='TVCS loss weight')
    parser.add_argument('--focal_gamma', type=float, default=1.0, help='Focal loss gamma for CK binary classifier')
    parser.add_argument('--residual_mu', type=float, default=0.01, help='Residual penalty weight')
    
    # Execution / Guardrails
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry-run check without training')
    parser.add_argument('--execute_train', action='store_true', help='Execute actual model training')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing output files')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("Stage F2: Train CIKD++ Residual TVCS-Transformer")
    print("=" * 80)

    # 1. Verify cache files exist
    required_cache_files = [
        'text_features.npy', 'image_features_global.npy', 'image_features_patch.npy',
        'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'y_ck.npy', 'split_ids.npy'
    ]
    cache_present = True
    for f in required_cache_files:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing required cache file: {p}")
            cache_present = False
        else:
            print(f"[+] Found cache file: {p}")

    if not cache_present:
        print("[-] Cache check failed.")
        sys.exit(1)

    # 2. Check baseline logits path
    baseline_logits_present = True
    baseline_files = ['train_logits_base.npy', 'val_logits_base.npy']
    for f in baseline_files:
        p = os.path.join(args.baseline_logits_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing baseline logits file: {p}")
            baseline_logits_present = False
        else:
            print(f"[+] Found baseline logits file: {p}")

    if not baseline_logits_present:
        if args.dry_run:
            print("[!] Warning: Baseline logits files are missing. Dry-run will use dummy logits.")
        else:
            print("[-] Baseline logits missing. Execute Stage F0 first.")
            sys.exit(1)

    # 3. Check TVCS specialist checkpoint path
    tvcs_checkpoint_present = os.path.exists(args.tvcs_checkpoint)
    if not tvcs_checkpoint_present:
        if args.dry_run:
            print(f"[!] Warning: TVCS Specialist checkpoint not found at: {args.tvcs_checkpoint}. Dry-run will use randomly initialized TVCS Specialist weights.")
        else:
            print(f"[-] TVCS Specialist checkpoint not found at: {args.tvcs_checkpoint}. Train the specialist first (Stage F1).")
            sys.exit(1)
    else:
        print(f"[+] Found TVCS Specialist checkpoint: {args.tvcs_checkpoint}")

    # Overwrite check
    planned_outputs = [
        args.checkpoint_out,
        os.path.join(args.out_dir, 'f2_rt_metrics_val.csv'),
        os.path.join(args.out_dir, 'f2_rt_summary.txt')
    ]
    if not args.overwrite:
        for path in planned_outputs:
            if os.path.exists(path):
                print(f"[-] Output file already exists: {path}")
                print("[-] Refusing to run or overwrite without the --overwrite flag.")
                sys.exit(1)

    # Load shapes from cache to verify
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    num_relations = int(np.load(os.path.join(args.cache_dir, 'relation_ids.npy'), mmap_mode='r').max()) + 1
    kg_dim = np.load(os.path.join(args.cache_dir, 'kg_features.npy'), mmap_mode='r').shape[1]

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Dry-run Mode
    if args.dry_run or not args.execute_train:
        print("\n--- DRY RUN STATUS ---")
        print("DRY RUN ONLY — no training executed.")
        
        # Instantiate model
        print("Instantiating CIKDPPResidualTransformer...")
        model = CIKDPPResidualTransformer(
            num_relations=num_relations,
            kg_dim=kg_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            alpha_init=args.alpha_init,
            alpha_max=args.alpha_max
        ).to(device)
        print("  Model successfully instantiated.")

        # Load TVCS checkpoint if available
        if tvcs_checkpoint_present:
            print("Loading TVCS Specialist checkpoint...")
            try:
                tvcs_ckpt = torch.load(args.tvcs_checkpoint, map_location=device, weights_only=False)
                state_dict = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
                model.tvcs_specialist.load_state_dict(state_dict)
                print("  TVCS Specialist weights loaded.")
            except Exception as e:
                print(f"  [-] Failed to load TVCS Specialist weights: {e}")

        # Forward smoke test
        print("Running forward pass smoke test with 2 samples...")
        text_dummy = torch.randn(2, 768).to(device)
        img_dummy = torch.randn(2, 512).to(device)
        patch_dummy = torch.randn(2, 49, 512).to(device)
        kg_dummy = torch.randn(2, kg_dim).to(device)
        rel_dummy = torch.zeros(2, dtype=torch.long).to(device)
        logits_base_dummy = torch.randn(2, 6).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(
                text_features=text_dummy,
                image_global_features=img_dummy,
                image_patch_features=patch_dummy,
                kg_features=kg_dummy,
                relation_ids=rel_dummy,
                baseline_logits=logits_base_dummy
            )
            
        print("  Forward pass output shapes:")
        for k, v in outputs.items():
            print(f"    {k}: {v.shape if hasattr(v, 'shape') else v}")

        print("\nPlanned output files:")
        for path in planned_outputs:
            print(f"  {path}")

        print("\n[+] DRY RUN COMPLETED SUCCESSFULLY.")
        print("\nNext command to run for future execution:")
        print(f"python src/stage_f2_train_residual_transformer.py --cache_dir {args.cache_dir} --baseline_logits_dir {args.baseline_logits_dir} --tvcs_checkpoint {args.tvcs_checkpoint} --out_dir {args.out_dir} --checkpoint_out {args.checkpoint_out} --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --weight_decay {args.weight_decay} --d_model {args.d_model} --num_layers {args.num_layers} --num_heads {args.num_heads} --dropout {args.dropout} --alpha_init {args.alpha_init} --alpha_max {args.alpha_max} --lambda_tvcs {args.lambda_tvcs} --focal_gamma {args.focal_gamma} --residual_mu {args.residual_mu} --patience {args.patience} --seed {args.seed} --execute_train" + (" --overwrite" if args.overwrite else ""))
        sys.exit(0)

    # 5. Future Execution Mode (Training)
    print("\nStarting model training...")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    text_feat = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    img_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    img_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    kg_feats = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    y_ck = np.load(os.path.join(args.cache_dir, 'y_ck.npy'))

    # Load baseline logits
    print("Loading baseline logits...")
    tr_logits_base = np.load(os.path.join(args.baseline_logits_dir, 'train_logits_base.npy'))
    val_logits_base = np.load(os.path.join(args.baseline_logits_dir, 'val_logits_base.npy'))

    # Extract Train/Val splits
    train_mask = (split_ids == 0)
    val_mask = (split_ids == 1)

    # Create tensors
    tr_text = torch.tensor(text_feat[train_mask], dtype=torch.float32)
    tr_img_g = torch.tensor(img_global[train_mask], dtype=torch.float32)
    tr_img_p = torch.tensor(img_patch[train_mask], dtype=torch.float32)
    tr_kg = torch.tensor(kg_feats[train_mask], dtype=torch.float32)
    tr_rel = torch.tensor(relation_ids[train_mask], dtype=torch.long)
    tr_lbl = torch.tensor(labels_fine[train_mask], dtype=torch.long)
    tr_y_ck = torch.tensor(y_ck[train_mask], dtype=torch.float32)
    tr_logits = torch.tensor(tr_logits_base, dtype=torch.float32)

    val_text = torch.tensor(text_feat[val_mask], dtype=torch.float32)
    val_img_g = torch.tensor(img_global[val_mask], dtype=torch.float32)
    val_img_p = torch.tensor(img_patch[val_mask], dtype=torch.float32)
    val_kg = torch.tensor(kg_feats[val_mask], dtype=torch.float32)
    val_rel = torch.tensor(relation_ids[val_mask], dtype=torch.long)
    val_lbl = torch.tensor(labels_fine[val_mask], dtype=torch.long)
    val_y_ck = torch.tensor(y_ck[val_mask], dtype=torch.float32)
    val_logits = torch.tensor(val_logits_base, dtype=torch.float32)

    # Dataloaders
    train_ds = TensorDataset(tr_text, tr_img_g, tr_img_p, tr_kg, tr_rel, tr_lbl, tr_y_ck, tr_logits)
    val_ds = TensorDataset(val_text, val_img_g, val_img_p, val_kg, val_rel, val_lbl, val_y_ck, val_logits)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Allocating CIKDPPResidualTransformer model on {device}...")
    model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        alpha_init=args.alpha_init,
        alpha_max=args.alpha_max
    ).to(device)

    # Load TVCS Specialist checkpoint (frozen TVCS Specialist option, or warm-start, let's load checkpoint and freeze/fine-tune it)
    print("Loading TVCS Specialist checkpoint...")
    tvcs_ckpt = torch.load(args.tvcs_checkpoint, map_location=device, weights_only=False)
    state_dict = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
    model.tvcs_specialist.load_state_dict(state_dict)
    
    # TVCS specialist can be frozen during training of the residual branch to act as a frozen anchor
    # Let's keep TVCS Specialist parameters frozen to act as specialist feature extractor
    for param in model.tvcs_specialist.parameters():
        param.requires_grad = False
    print("  [+] TVCS Specialist parameters frozen.")

    # Calculate class-weighted CE loss for classification
    counts = np.bincount(labels_fine[train_mask], minlength=6)
    counts = np.maximum(counts, 1)
    weights = len(tr_lbl) / (6.0 * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Calculated class weights: {weights.tolist()}")

    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_tvcs = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_cls_loss = 0.0
        train_tvcs_loss = 0.0
        train_res_loss = 0.0
        train_ck_loss = 0.0
        train_total_loss = 0.0

        for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_y_ck, bx_logits in train_loader:
            bx_text = bx_text.to(device)
            bx_img_g = bx_img_g.to(device)
            bx_img_p = bx_img_p.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_lbl = bx_lbl.to(device)
            bx_y_ck = bx_y_ck.to(device)
            bx_logits = bx_logits.to(device)

            optimizer.zero_grad()
            outputs = model(bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits)

            logits_final = outputs['logits_final']
            logits_delta = outputs['logits_delta']
            c_logit = outputs['c_logit']

            # 1. Main classification loss
            loss_cls = criterion_cls(logits_final, bx_lbl)

            # 2. TVCS loss
            mask_tvcs = (bx_y_ck != -1)
            if mask_tvcs.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logit[mask_tvcs], bx_y_ck[mask_tvcs])
            else:
                loss_tvcs = torch.tensor(0.0, device=device)

            # 3. Residual L2 penalty
            loss_res = torch.mean(logits_delta ** 2)

            # 4. Focal binary loss for CK class (class 2)
            target_ck = (bx_lbl == 2).float()
            pred_logit_ck = logits_final[:, 2]
            bce_loss_ck = nn.functional.binary_cross_entropy_with_logits(pred_logit_ck, target_ck, reduction='none')
            probs_ck = torch.sigmoid(pred_logit_ck)
            p_t = probs_ck * target_ck + (1.0 - probs_ck) * (1.0 - target_ck)
            focal_weight = (1.0 - p_t) ** args.focal_gamma
            loss_ck_binary = focal_weight.mean() * bce_loss_ck.mean()

            # Total loss
            loss_total = loss_cls + args.lambda_tvcs * loss_tvcs + args.residual_mu * loss_res + 0.2 * loss_ck_binary

            loss_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_cls_loss += loss_cls.item() * len(bx_lbl)
            train_tvcs_loss += loss_tvcs.item() * mask_tvcs.sum().item()
            train_res_loss += loss_res.item() * len(bx_lbl)
            train_ck_loss += loss_ck_binary.item() * len(bx_lbl)
            train_total_loss += loss_total.item() * len(bx_lbl)

        train_cls_loss /= len(tr_lbl)
        train_res_loss /= len(tr_lbl)
        train_ck_loss /= len(tr_lbl)
        train_total_loss /= len(tr_lbl)

        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []
        val_c_probs = []
        val_y_ck_list = []

        with torch.no_grad():
            for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_y_ck, bx_logits in val_loader:
                bx_text = bx_text.to(device)
                bx_img_g = bx_img_g.to(device)
                bx_img_p = bx_img_p.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)
                bx_logits = bx_logits.to(device)

                outputs = model(bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits)
                logits_final = outputs['logits_final']
                c_logit = outputs['c_logit']

                preds = torch.argmax(logits_final, dim=-1).cpu().numpy()
                probs_tvcs = torch.sigmoid(c_logit).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(bx_lbl.cpu().numpy())
                val_c_probs.extend(probs_tvcs)
                val_y_ck_list.extend(bx_y_ck.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_c_probs = np.array(val_c_probs)
        val_y_ck_arr = np.array(val_y_ck_list)

        acc = accuracy_score(val_targets, val_preds)
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(6)), zero_division=0)
        ck_f1 = per_class_f1[2]

        tvcs_mask = (val_y_ck_arr != -1)
        if tvcs_mask.sum() > 0 and len(np.unique(val_y_ck_arr[tvcs_mask])) > 1:
            tvcs_auc = roc_auc_score(val_y_ck_arr[tvcs_mask], val_c_probs[tvcs_mask])
        else:
            tvcs_auc = 0.5

        # Compute validation selection score: 0.45 * Macro-F1 + 0.35 * CK-F1 + 0.20 * TVCS_AUC
        val_score = 0.45 * macro_f1 + 0.35 * ck_f1 + 0.20 * tvcs_auc

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Loss: {train_total_loss:.4f} | Val Acc: {acc:.4f} | Val Macro-F1: {macro_f1:.4f} | Val CK-F1: {ck_f1:.4f} | Val TVCS AUC: {tvcs_auc:.4f} | Score: {val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'val_metrics': {
                    'accuracy': acc,
                    'macro_f1': macro_f1,
                    'ck_f1': ck_f1,
                    'tvcs_auc': tvcs_auc,
                    'per_class_f1': per_class_f1.tolist()
                }
            }
            torch.save(checkpoint, args.checkpoint_out)
            print(f"  [+] Saved best checkpoint with Val Score {val_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered. Best epoch was {best_epoch} with Val Score {best_score:.4f}")
                break

    # Load best checkpoint and write final outputs
    print(f"\nLoading best checkpoint from {args.checkpoint_out}...")
    checkpoint = torch.load(args.checkpoint_out, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Save metrics CSV
    metrics_path = os.path.join(args.out_dir, 'f2_rt_metrics_val.csv')
    best_m = checkpoint['val_metrics']
    df_metrics = pd.DataFrame([{
        'model': 'cikd_pp_rt',
        'accuracy': best_m['accuracy'],
        'macro_f1': best_m['macro_f1'],
        'ck_f1': best_m['ck_f1'],
        'tvcs_auc': best_m['tvcs_auc'],
        'selection_score': checkpoint['val_score'],
        'best_epoch': checkpoint['epoch']
    }])
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[+] Saved validation metrics to {metrics_path}")

    # Save summary text
    summary_path = os.path.join(args.out_dir, 'f2_rt_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CIKD++ Residual TVCS-Transformer Training Summary\n")
        f.write("===============================================\n")
        f.write(f"Best Epoch: {checkpoint['epoch']}\n")
        f.write(f"Validation Selection Score: {checkpoint['val_score']:.4f}\n")
        f.write(f"Validation Accuracy: {best_m['accuracy']:.4f}\n")
        f.write(f"Validation Macro-F1: {best_m['macro_f1']:.4f}\n")
        f.write(f"Validation CK-F1: {best_m['ck_f1']:.4f}\n")
        f.write(f"Validation TVCS AUC: {best_m['tvcs_auc']:.4f}\n")
        f.write(f"Per-Class F1: {best_m['per_class_f1']}\n")
    print(f"[+] Saved summary to {summary_path}")

    print("\n[+] MODEL TRAINING AND EXPORT COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
