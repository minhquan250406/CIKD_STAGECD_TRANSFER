"""
Stage F0: Export Baseline Logits.
Prepares baseline logits and probabilities from the text_image_kg_concat baseline model.
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
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SimpleMLP(nn.Module):
    """
    Simple MLP Classifier.
    Matching run_stage_cd.py's SimpleMLP.
    """
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

def main():
    parser = argparse.ArgumentParser(description="Export baseline model logits and probabilities")
    parser.add_argument('--cache_dir', type=str, default='data/cache/kg_complete', help='Cache directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/baselines/text_image_kg_concat_seed42.pt', help='Baseline model checkpoint')
    parser.add_argument('--out_dir', type=str, default='outputs/stage_f0_baseline_anchor', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Inference batch size')
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry-run check without inference')
    parser.add_argument('--execute', action='store_true', help='Execute actual inference and save results')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing output files')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("Stage F0: Export Baseline Logits")
    print("=" * 80)

    # 1. Check required files exist
    required_features = ['text_features.npy', 'image_features_global.npy', 'kg_features.npy', 'split_ids.npy', 'labels_fine.npy']
    features_present = True
    print("Checking cache files...")
    for f in required_features:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing required cache file: {p}")
            features_present = False
        else:
            print(f"[+] Found cache file: {p}")

    if not features_present:
        print("[-] Cache check failed. Ensure features are pre-extracted.")
        sys.exit(1)

    checkpoint_present = os.path.exists(args.checkpoint)
    if not checkpoint_present:
        print(f"[-] Baseline checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    else:
        print(f"[+] Found baseline checkpoint: {args.checkpoint}")

    # Define planned output files
    output_files = [
        'train_logits_base.npy', 'val_logits_base.npy', 'test_logits_base.npy',
        'train_probs_base.npy', 'val_probs_base.npy', 'test_probs_base.npy',
        'f0_baseline_anchor_metrics.csv', 'F0_README.txt'
    ]
    
    # 2. Overwrite check
    if not args.overwrite:
        for f in output_files:
            p = os.path.join(args.out_dir, f)
            if os.path.exists(p):
                print(f"[-] Output file already exists: {p}")
                print("[-] Refusing to run or overwrite without the --overwrite flag.")
                sys.exit(1)

    # 3. Dry-run Mode
    if args.dry_run or not args.execute:
        print("\n--- DRY RUN STATUS ---")
        # Load shapes only (using memory mapping)
        text_shape = np.load(os.path.join(args.cache_dir, 'text_features.npy'), mmap_mode='r').shape
        img_shape = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'), mmap_mode='r').shape
        kg_shape = np.load(os.path.join(args.cache_dir, 'kg_features.npy'), mmap_mode='r').shape
        split_shape = np.load(os.path.join(args.cache_dir, 'split_ids.npy'), mmap_mode='r').shape
        labels_shape = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'), mmap_mode='r').shape

        print(f"Cache Shapes:")
        print(f"  text_features: {text_shape}")
        print(f"  image_features_global: {img_shape}")
        print(f"  kg_features: {kg_shape}")
        print(f"  split_ids: {split_shape}")
        print(f"  labels_fine: {labels_shape}")

        # Safely inspect checkpoint
        print("Inspecting checkpoint keys safely on CPU...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("  Found 'model_state_dict'")
        else:
            state_dict = checkpoint
            print("  Using raw state_dict")
        print(f"  Number of model layers: {len(state_dict)}")

        print("\nExpected output paths:")
        for f in output_files:
            print(f"  {os.path.join(args.out_dir, f)}")

        print("\n[+] DRY RUN COMPLETED SUCCESSFULLY.")
        print("\nNext command to run for future execution:")
        print(f"python src/stage_f0_export_baseline_logits.py --cache_dir {args.cache_dir} --checkpoint {args.checkpoint} --out_dir {args.out_dir} --batch_size {args.batch_size} --seed {args.seed} --execute" + (" --overwrite" if args.overwrite else ""))
        sys.exit(0)

    # 4. Future Execution Mode
    print("\nStarting baseline logits extraction...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load feature arrays
    print("Loading feature arrays...")
    text_feat = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    img_feat = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    kg_feat = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))

    # Concatenate features for SimpleMLP
    features = np.concatenate([text_feat, img_feat, kg_feat], axis=1)
    input_dim = features.shape[1]
    print(f"Concatenated feature shape: {features.shape} (dim: {input_dim})")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = SimpleMLP(input_dim=input_dim, num_classes=6).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    splits = {
        'train': (split_ids == 0),
        'val': (split_ids == 1),
        'test': (split_ids == 2)
    }

    metrics_records = []

    for split_name, mask in splits.items():
        sub_feats = features[mask]
        sub_labels = labels_fine[mask]
        
        if len(sub_feats) == 0:
            print(f"[-] No samples found for split: {split_name}")
            continue

        dataset = TensorDataset(torch.tensor(sub_feats, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_logits = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(device)
                logits = model(bx)
                all_logits.append(logits.cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        probs_np = torch.softmax(torch.tensor(logits_np), dim=-1).numpy()
        preds_np = np.argmax(logits_np, axis=1)

        # Save files
        logits_path = os.path.join(args.out_dir, f"{split_name}_logits_base.npy")
        probs_path = os.path.join(args.out_dir, f"{split_name}_probs_base.npy")
        np.save(logits_path, logits_np)
        np.save(probs_path, probs_np)
        print(f"[+] Saved logits to {logits_path}")
        print(f"[+] Saved probabilities to {probs_path}")

        # Compute metrics
        acc = accuracy_score(sub_labels, preds_np)
        macro_f1 = f1_score(sub_labels, preds_np, average='macro', zero_division=0)
        weighted_f1 = f1_score(sub_labels, preds_np, average='weighted', zero_division=0)
        per_class_f1 = f1_score(sub_labels, preds_np, average=None, labels=list(range(6)), zero_division=0)
        ck_f1 = per_class_f1[2]

        metrics_records.append({
            'split': split_name,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'ck_f1': ck_f1
        })

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(args.out_dir, 'f0_baseline_anchor_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[+] Saved metrics CSV to {metrics_path}")

    # Save README
    readme_path = os.path.join(args.out_dir, 'F0_README.txt')
    with open(readme_path, 'w') as f:
        f.write("Baseline Anchor Logits and Probabilities Exported.\n")
        f.write(f"Source Checkpoint: {args.checkpoint}\n")
        f.write(f"Source Cache: {args.cache_dir}\n")
        f.write(f"Seed: {args.seed}\n")
    print(f"[+] Saved README to {readme_path}")

    print("\n[+] LOGITS EXPORT COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
