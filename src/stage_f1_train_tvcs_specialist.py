"""
Stage F1: Train TVCS Specialist.
Prepares code to train the TVCS Specialist on the tvcs_eligible cached subset.
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
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# Import model
from models.cikd_pp_rt import TVCSSpecialist

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train TVCS Specialist model")
    parser.add_argument('--cache_dir', type=str, default='data/cache/tvcs_eligible', help='Cache directory')
    parser.add_argument('--out_dir', type=str, default='outputs/stage_f1_tvcs_specialist', help='Output directory')
    parser.add_argument('--checkpoint_out', type=str, default='checkpoints/stage_f/tvcs_specialist_seed42.pt', help='Path to save trained checkpoint')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry-run check without training')
    parser.add_argument('--execute_train', action='store_true', help='Execute actual model training')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing output files')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("Stage F1: Train TVCS Specialist")
    print("=" * 80)

    # 1. Verify cache files exist
    required_files = [
        'kg_features.npy', 'relation_ids.npy', 'image_features_patch.npy',
        'y_ck.npy', 'split_ids.npy', 'sample_ids.npy'
    ]
    cache_present = True
    for f in required_files:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing required cache file: {p}")
            cache_present = False
        else:
            print(f"[+] Found cache file: {p}")

    if not cache_present:
        print("[-] Cache check failed.")
        sys.exit(1)

    # Define planned outputs
    planned_outputs = [
        args.checkpoint_out,
        os.path.join(args.out_dir, 'f1_tvcs_metrics_val.csv'),
        os.path.join(args.out_dir, 'f1_tvcs_scores_val.csv'),
        os.path.join(args.out_dir, 'f1_tvcs_hist_val.png'),
        os.path.join(args.out_dir, 'f1_tvcs_summary.txt')
    ]

    # Overwrite check
    if not args.overwrite:
        for path in planned_outputs:
            if os.path.exists(path):
                print(f"[-] Output file already exists: {path}")
                print("[-] Refusing to run or overwrite without the --overwrite flag.")
                sys.exit(1)

    # Load y_ck and split_ids to verify shapes/splits
    print("Checking y_ck labels and split IDs...")
    y_ck = np.load(os.path.join(args.cache_dir, 'y_ck.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))

    # Verify y_ck labels contain only real/CK-compatible values (0 and 1) for TVCS training
    # Check unique values excluding any potential -1 padding/ineligibility
    unique_y_ck = np.unique(y_ck)
    print(f"  Unique y_ck values in cache: {unique_y_ck.tolist()}")
    
    # Check if we have invalid values (values other than 0, 1, and possibly -1)
    invalid_mask = ~np.isin(y_ck, [0, 1, -1])
    if invalid_mask.sum() > 0:
        print(f"[-] Error: y_ck contains invalid labels: {np.unique(y_ck[invalid_mask]).tolist()}")
        sys.exit(1)
    
    # We want to train on y_ck != -1 (meaning eligible TVCS samples)
    eligible_mask = (y_ck != -1)
    eligible_count = int(np.sum(eligible_mask))
    print(f"  Total eligible TVCS samples (y_ck != -1): {eligible_count} out of {len(y_ck)}")

    train_count = int(np.sum((split_ids == 0) & eligible_mask))
    val_count = int(np.sum((split_ids == 1) & eligible_mask))
    test_count = int(np.sum((split_ids == 2) & eligible_mask))

    print(f"  Split Counts (Eligible TVCS):")
    print(f"    Train: {train_count}")
    print(f"    Val: {val_count}")
    print(f"    Test: {test_count}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load relation_ids max value to configure embedding vocab size
    relation_ids_cached = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    num_relations = int(relation_ids_cached.max()) + 1
    kg_dim = np.load(os.path.join(args.cache_dir, 'kg_features.npy'), mmap_mode='r').shape[1]
    
    # 2. Dry-run Mode
    if args.dry_run or not args.execute_train:
        print("\n--- DRY RUN STATUS ---")
        print("DRY RUN ONLY — no training executed.")
        
        # Instantiate model
        print("Instantiating TVCSSpecialist model...")
        model = TVCSSpecialist(
            num_relations=num_relations,
            kg_dim=kg_dim,
            relation_emb_dim=64,
            tvcs_dim=512,
            image_patch_dim=512,
            c_emb_dim=64
        ).to(device)
        print(f"  TVCSSpecialist structure loaded on {device}.")

        # Perform forward-shape smoke test using a tiny dummy batch of shape
        print("Running single forward-shape smoke test on a tiny batch...")
        kg_dummy = torch.randn(2, kg_dim).to(device)
        rel_dummy = torch.zeros(2, dtype=torch.long).to(device)
        patch_dummy = torch.randn(2, 49, 512).to(device)
        
        model.eval()
        with torch.no_grad():
            z_v, c_logit, c_emb, attention_weights = model(kg_dummy, rel_dummy, patch_dummy)
            
        print(f"  Forward-pass outputs check:")
        print(f"    z_v shape: {z_v.shape} (Expected: [2, 512])")
        print(f"    c_logit shape: {c_logit.shape} (Expected: [2])")
        print(f"    c_emb shape: {c_emb.shape} (Expected: [2, 64])")
        print(f"    attention_weights shape: {attention_weights.shape} (Expected: [2, 49])")

        print("\nPlanned output files:")
        for path in planned_outputs:
            print(f"  {path}")

        print("\n[+] DRY RUN COMPLETED SUCCESSFULLY.")
        print("\nNext command to run for future execution:")
        print(f"python src/stage_f1_train_tvcs_specialist.py --cache_dir {args.cache_dir} --out_dir {args.out_dir} --checkpoint_out {args.checkpoint_out} --epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} --weight_decay {args.weight_decay} --patience {args.patience} --seed {args.seed} --execute_train" + (" --overwrite" if args.overwrite else ""))
        sys.exit(0)

    # 3. Future Training execution
    print("\nStarting model training...")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)

    # Load arrays
    print("Loading datasets...")
    kg_feats = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    image_patches = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    sample_ids = np.load(os.path.join(args.cache_dir, 'sample_ids.npy'))

    # Extract Train/Val splits filtering out ineligible samples
    train_mask = (split_ids == 0) & eligible_mask
    val_mask = (split_ids == 1) & eligible_mask

    # Train tensors
    tr_kg = torch.tensor(kg_feats[train_mask], dtype=torch.float32)
    tr_rel = torch.tensor(relation_ids[train_mask], dtype=torch.long)
    tr_patch = torch.tensor(image_patches[train_mask], dtype=torch.float32)
    tr_y = torch.tensor(y_ck[train_mask], dtype=torch.float32)

    # Val tensors
    val_kg = torch.tensor(kg_feats[val_mask], dtype=torch.float32)
    val_rel = torch.tensor(relation_ids[val_mask], dtype=torch.long)
    val_patch = torch.tensor(image_patches[val_mask], dtype=torch.float32)
    val_y = torch.tensor(y_ck[val_mask], dtype=torch.float32)
    val_sid = torch.tensor(sample_ids[val_mask], dtype=torch.long)

    # Dataloaders
    train_ds = TensorDataset(tr_kg, tr_rel, tr_patch, tr_y)
    val_ds = TensorDataset(val_kg, val_rel, val_patch, val_y, val_sid)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Allocating TVCSSpecialist on {device}...")
    model = TVCSSpecialist(
        num_relations=num_relations,
        kg_dim=kg_dim,
        relation_emb_dim=64,
        tvcs_dim=512,
        image_patch_dim=512,
        c_emb_dim=64
    ).to(device)

    # Compute binary class weights to handle imbalance in y_ck
    tr_y_np = tr_y.numpy()
    neg_count = np.sum(tr_y_np == 0)
    pos_count = np.sum(tr_y_np == 1)
    print(f"Training Class distribution: Consistent (y_ck=0): {neg_count}, Contradictory (y_ck=1): {pos_count}")
    pos_weight_val = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    print(f"Calculated positive class BCE loss weight: {pos_weight_val:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for bx_kg, bx_rel, bx_patch, bx_y in train_loader:
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_patch = bx_patch.to(device)
            bx_y = bx_y.to(device)

            optimizer.zero_grad()
            _, c_logit, _, _ = model(bx_kg, bx_rel, bx_patch)
            loss = criterion(c_logit, bx_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx_y)

        train_loss /= len(tr_y)

        # Validation
        model.eval()
        val_loss = 0.0
        val_logits_all = []
        val_y_all = []
        with torch.no_grad():
            for bx_kg, bx_rel, bx_patch, bx_y, _ in val_loader:
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)
                bx_patch = bx_patch.to(device)
                bx_y = bx_y.to(device)

                _, c_logit, _, _ = model(bx_kg, bx_rel, bx_patch)
                loss = criterion(c_logit, bx_y)
                val_loss += loss.item() * len(bx_y)
                
                val_logits_all.extend(c_logit.cpu().numpy())
                val_y_all.extend(bx_y.cpu().numpy())

        val_loss /= len(val_y)
        val_logits_all = np.array(val_logits_all)
        val_probs_all = 1.0 / (1.0 + np.exp(-val_logits_all))
        val_y_all = np.array(val_y_all)

        # Compute validation AUC and PR-AUC
        if len(np.unique(val_y_all)) > 1:
            val_auc = roc_auc_score(val_y_all, val_probs_all)
            val_pr_auc = average_precision_score(val_y_all, val_probs_all)
        else:
            val_auc = 0.5
            val_pr_auc = 0.0

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_pr_auc': val_pr_auc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, args.checkpoint_out)
            print(f"  [+] Saved best checkpoint with Val AUC {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered. Best epoch was {best_epoch} with Val AUC {best_val_auc:.4f}")
                break

    # Load best model for final validation outputs
    print(f"\nLoading best model checkpoint from {args.checkpoint_out} for saving validation outputs...")
    checkpoint = torch.load(args.checkpoint_out, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_logits_all = []
    val_probs_all = []
    val_y_all = []
    val_sid_all = []

    with torch.no_grad():
        for bx_kg, bx_rel, bx_patch, bx_y, bx_sid in val_loader:
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_patch = bx_patch.to(device)
            
            _, c_logit, _, _ = model(bx_kg, bx_rel, bx_patch)
            c_prob = torch.sigmoid(c_logit)
            
            val_logits_all.extend(c_logit.cpu().numpy())
            val_probs_all.extend(c_prob.cpu().numpy())
            val_y_all.extend(bx_y.numpy())
            val_sid_all.extend(bx_sid.numpy())

    val_logits_all = np.array(val_logits_all)
    val_probs_all = np.array(val_probs_all)
    val_y_all = np.array(val_y_all)
    val_sid_all = np.array(val_sid_all)

    # Compute final metrics
    val_auc = roc_auc_score(val_y_all, val_probs_all)
    val_pr_auc = average_precision_score(val_y_all, val_probs_all)
    mean_c_real = float(np.mean(val_probs_all[val_y_all == 0]))
    mean_c_ck = float(np.mean(val_probs_all[val_y_all == 1]))
    tvcs_delta = mean_c_ck - mean_c_real

    # Save metrics CSV
    metrics_path = os.path.join(args.out_dir, 'f1_tvcs_metrics_val.csv')
    df_metrics = pd.DataFrame([{
        'model': 'tvcs_specialist',
        'val_auc': val_auc,
        'val_pr_auc': val_pr_auc,
        'mean_c_real': mean_c_real,
        'mean_c_ck': mean_c_ck,
        'tvcs_delta': tvcs_delta
    }])
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[+] Saved validation metrics to {metrics_path}")

    # Save scores CSV
    scores_path = os.path.join(args.out_dir, 'f1_tvcs_scores_val.csv')
    df_scores = pd.DataFrame({
        'sample_id': val_sid_all,
        'y_ck': val_y_all,
        'c_logit': val_logits_all,
        'c_score': val_probs_all
    })
    df_scores.to_csv(scores_path, index=False)
    print(f"[+] Saved validation contradiction scores to {scores_path}")

    # Save histogram
    plt.figure(figsize=(8, 6))
    plt.hist(val_probs_all[val_y_all == 0], bins=30, alpha=0.5, label=f'Real (y_ck=0, mean={mean_c_real:.4f})', color='green', edgecolor='k')
    plt.hist(val_probs_all[val_y_all == 1], bins=30, alpha=0.5, label=f'Contradictory (y_ck=1, mean={mean_c_ck:.4f})', color='red', edgecolor='k')
    plt.xlabel('Contradiction Score')
    plt.ylabel('Count')
    plt.title(f'TVCS Specialist Scores Distribution (Val, Delta={tvcs_delta:.4f})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    hist_path = os.path.join(args.out_dir, 'f1_tvcs_hist_val.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved histogram plot to {hist_path}")

    # Save summary text
    summary_path = os.path.join(args.out_dir, 'f1_tvcs_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TVCS Specialist Training Summary\n")
        f.write("===============================\n")
        f.write(f"Validation AUC: {val_auc:.4f}\n")
        f.write(f"Validation PR-AUC: {val_pr_auc:.4f}\n")
        f.write(f"Mean Contradiction Score (Consistent): {mean_c_real:.4f}\n")
        f.write(f"Mean Contradiction Score (Contradictory): {mean_c_ck:.4f}\n")
        f.write(f"TVCS Delta: {tvcs_delta:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
    print(f"[+] Saved text summary to {summary_path}")

    print("\n[+] MODEL TRAINING AND EXPORT COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
