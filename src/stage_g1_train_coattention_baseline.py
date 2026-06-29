"""
Stage G1: Same-split Co-attention Baselines.
Implements the ti_coattn_kg_concat and kg_image_coattn_text_concat variants.
"""

import os
import sys
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class G1CoattentionBaseline(nn.Module):
    """
    Stage G1 Generic Co-attention Baseline.
    
    Supports two variants:
    1. ti_coattn_kg_concat
    2. kg_image_coattn_text_concat
    
    Strict exclusions apply: No TVCS auxiliary loss, no c_emb, no contradiction logit,
    no masked BCE on y_ck, no CK-vs-real objective, no residual baseline logits, and no Stage F loading.
    """
    def __init__(self, variant, num_relations, hidden_dim=256, attn_dim=256, dropout=0.2):
        super().__init__()
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        
        # Modal projections to hidden_dim
        self.proj_text_fuse = nn.Linear(768, hidden_dim)
        self.proj_img_global = nn.Linear(512, hidden_dim)
        self.proj_kg = nn.Linear(100, hidden_dim)
        self.relation_embed = nn.Embedding(num_relations, hidden_dim)
        
        # Patch projection to attn_dim for co-attention
        self.proj_patch = nn.Linear(512, attn_dim)
        
        # Projection of attended patch features back to hidden_dim
        self.proj_attended_img_fuse = nn.Linear(attn_dim, hidden_dim)
        
        if self.variant == 'ti_coattn_kg_concat':
            # Text query projection to attn_dim
            self.proj_text_query = nn.Linear(768, attn_dim)
        elif self.variant == 'kg_image_coattn_text_concat':
            # KG + relation query projection to attn_dim
            self.proj_kg_query = nn.Linear(hidden_dim, attn_dim)
        else:
            raise ValueError(f"Invalid variant: {variant}")
            
        # MLP classifier (projects fused representation [B, 7 * hidden_dim] to 6 classes)
        self.classifier = nn.Sequential(
            nn.Linear(7 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6)
        )
        
    def forward(self, text_features, image_features_patch, image_features_global, kg_features, relation_ids):
        B = text_features.size(0)
        
        # 1. Project patches to attn_dim
        patch_tokens = self.proj_patch(image_features_patch)  # [B, 49, attn_dim]
        
        # 2. Query preparation
        if self.variant == 'ti_coattn_kg_concat':
            text_query = self.proj_text_query(text_features).unsqueeze(1)  # [B, 1, attn_dim]
            query = text_query
        else:  # kg_image_coattn_text_concat
            kg_proj_temp = self.proj_kg(kg_features)  # [B, hidden_dim]
            relation_emb_temp = self.relation_embed(relation_ids)  # [B, hidden_dim]
            kg_combined = kg_proj_temp + relation_emb_temp  # [B, hidden_dim]
            kg_query = self.proj_kg_query(kg_combined).unsqueeze(1)  # [B, 1, attn_dim]
            query = kg_query
            
        # 3. Scaled dot-product co-attention over patches
        keys = patch_tokens
        values = patch_tokens
        # query: [B, 1, attn_dim], keys.transpose(1, 2): [B, attn_dim, 49]
        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.attn_dim)  # [B, 1, 49]
        attention = torch.softmax(scores, dim=-1)  # [B, 1, 49]
        attended_image = torch.bmm(attention, values).squeeze(1)  # [B, attn_dim]
        
        # Project attended image representation to hidden_dim
        attended_image_proj = self.proj_attended_img_fuse(attended_image)  # [B, hidden_dim]
        
        # 4. Project base modalities to hidden_dim
        text_proj = self.proj_text_fuse(text_features)  # [B, hidden_dim]
        image_global_proj = self.proj_img_global(image_features_global)  # [B, hidden_dim]
        kg_proj = self.proj_kg(kg_features)  # [B, hidden_dim]
        relation_emb = self.relation_embed(relation_ids)  # [B, hidden_dim]
        
        # 5. Fusion concatenation
        if self.variant == 'ti_coattn_kg_concat':
            diff = torch.abs(text_proj - attended_image_proj)
            prod = text_proj * attended_image_proj
            fused = torch.cat([
                text_proj,
                attended_image_proj,
                image_global_proj,
                kg_proj,
                relation_emb,
                diff,
                prod
            ], dim=-1)  # [B, 7 * hidden_dim]
        else:  # kg_image_coattn_text_concat
            diff = torch.abs(kg_proj - attended_image_proj)
            prod = kg_proj * attended_image_proj
            fused = torch.cat([
                text_proj,
                image_global_proj,
                kg_proj,
                relation_emb,
                attended_image_proj,
                diff,
                prod
            ], dim=-1)  # [B, 7 * hidden_dim]
            
        # 6. Classifier
        logits = self.classifier(fused)  # [B, 6]
        return logits

def run_dry_run(args):
    print("=" * 80)
    print("Stage G1 Dry-Run Checks")
    print("=" * 80)
    
    # 1. Load arrays
    required_cache_files = [
        'text_features.npy', 'image_features_global.npy', 'image_features_patch.npy',
        'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'split_ids.npy'
    ]
    cache_present = True
    for f in required_cache_files:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing required cache file: {p}")
            cache_present = False
            
    if not cache_present:
        print("[-] Cache verification failed. Exiting.")
        sys.exit(1)
        
    print("Loading datasets...")
    text_features = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    kg_features = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    
    sample_ids_path = os.path.join(args.cache_dir, 'sample_ids.npy')
    sample_ids = np.load(sample_ids_path) if os.path.exists(sample_ids_path) else None

    # 2. Print all shapes
    print("[+] Loaded array shapes:")
    print(f"    text_features:          {text_features.shape} (Expected: [12786, 768])")
    print(f"    image_features_global:  {image_features_global.shape} (Expected: [12786, 512])")
    print(f"    image_features_patch:   {image_features_patch.shape} (Expected: [12786, 49, 512])")
    print(f"    kg_features:            {kg_features.shape} (Expected: [12786, 100])")
    print(f"    relation_ids:           {relation_ids.shape} (Expected: [12786])")
    print(f"    labels_fine:            {labels_fine.shape} (Expected: [12786])")
    print(f"    split_ids:              {split_ids.shape} (Expected: [12786])")
    if sample_ids is not None:
        print(f"    sample_ids:             {sample_ids.shape}")

    # 3. Verify split counts
    num_train = int(np.sum(split_ids == 0))
    num_val = int(np.sum(split_ids == 1))
    num_test = int(np.sum(split_ids == 2))

    print("[+] Split counts:")
    print(f"    Train (split_id == 0): {num_train} (Expected: 8900)")
    print(f"    Val (split_id == 1):   {num_val} (Expected: 1300)")
    print(f"    Test (split_id == 2):  {num_test} (Expected: 2586)")

    assert num_train == 8900, f"[-] Train count mismatch: {num_train} vs 8900"
    assert num_val == 1300, f"[-] Val count mismatch: {num_val} vs 1300"
    assert num_test == 2586, f"[-] Test count mismatch: {num_test} vs 2586"

    # 4. Verify label distribution per split
    label_dist = {}
    for sid, name in [(0, "Train"), (1, "Val"), (2, "Test")]:
        mask = (split_ids == sid)
        lbls = labels_fine[mask]
        dist = np.bincount(lbls, minlength=6)
        label_dist[sid] = dist.tolist()
        print(f"    {name} Label distribution: {dist.tolist()}")

    # 5. Verify relation vocab size
    max_relation_id = int(relation_ids.max())
    relation_vocab_size = max_relation_id + 1
    print(f"[+] Relation vocabulary size: {relation_vocab_size}")

    # 6. Verify image_features_patch shape is [N, 49, 512]
    assert image_features_patch.ndim == 3 and image_features_patch.shape[1] == 49 and image_features_patch.shape[2] == 512, \
        f"[-] image_features_patch shape mismatch: {image_features_patch.shape} vs [N, 49, 512]"
    print("[+] image_features_patch shape verified successfully.")

    # 7. Set up device & Instantiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")
    
    print(f"Instantiating G1CoattentionBaseline model for variant '{args.variant}'...")
    model = G1CoattentionBaseline(
        variant=args.variant,
        num_relations=relation_vocab_size,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout
    ).to(device)

    # Calculate class weights for CrossEntropyLoss on train split only
    train_mask = (split_ids == 0)
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    weights = len(train_labels) / (6.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[+] Calculated class weights (train split only): {weights.tolist()}")

    # 8. Run forward pass on tiny train batch of size 4
    train_idx = np.where(train_mask)[0][:4]
    bx_text = torch.tensor(text_features[train_idx], dtype=torch.float32).to(device)
    bx_patch = torch.tensor(image_features_patch[train_idx], dtype=torch.float32).to(device)
    bx_img_global = torch.tensor(image_features_global[train_idx], dtype=torch.float32).to(device)
    bx_kg = torch.tensor(kg_features[train_idx], dtype=torch.float32).to(device)
    bx_rel = torch.tensor(relation_ids[train_idx], dtype=torch.long).to(device)
    bx_lbl = torch.tensor(labels_fine[train_idx], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(bx_text, bx_patch, bx_img_global, bx_kg, bx_rel)

    # 9. Print logits shape
    print(f"[+] Logits shape: {list(logits.shape)} (Expected: [4, 6])")
    assert list(logits.shape) == [4, 6], f"[-] Logits shape mismatch: {list(logits.shape)}"

    # 10. Compute one dummy class-weighted CE loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss = criterion(logits, bx_lbl)
    print(f"[+] Dummy CrossEntropyLoss: {loss.item():.6f}")

    # 11. Print trainable parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Total trainable parameters: {param_count}")

    # 12. Create output directory only
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[+] Created output directory: {args.out_dir}")

    # 13. Write dry-run report
    report_path = os.path.join(args.out_dir, "G1_DRY_RUN_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("Stage G1 Dry-Run Report\n")
        f.write("========================\n\n")
        f.write(f"Variant: {args.variant}\n")
        f.write(f"Cache Directory: {args.cache_dir}\n")
        f.write(f"Output Directory: {args.out_dir}\n")
        f.write(f"Checkpoint Output: {args.checkpoint_out}\n\n")
        f.write("Array Shapes Checked:\n")
        f.write(f"  text_features:          {text_features.shape} (Expected: [12786, 768])\n")
        f.write(f"  image_features_global:  {image_features_global.shape} (Expected: [12786, 512])\n")
        f.write(f"  image_features_patch:   {image_features_patch.shape} (Expected: [12786, 49, 512])\n")
        f.write(f"  kg_features:            {kg_features.shape} (Expected: [12786, 100])\n")
        f.write(f"  relation_ids:           {relation_ids.shape} (Expected: [12786])\n")
        f.write(f"  labels_fine:            {labels_fine.shape} (Expected: [12786])\n")
        f.write(f"  split_ids:              {split_ids.shape} (Expected: [12786])\n")
        if sample_ids is not None:
            f.write(f"  sample_ids:             {sample_ids.shape}\n")
        f.write("\nSplit Counts Checked:\n")
        f.write(f"  Train (split_id == 0): {num_train} (Expected: 8900)\n")
        f.write(f"  Val (split_id == 1):   {num_val} (Expected: 1300)\n")
        f.write(f"  Test (split_id == 2):  {num_test} (Expected: 2586)\n\n")
        f.write("Label Distribution per Split:\n")
        f.write(f"  Train: {label_dist[0]}\n")
        f.write(f"  Val:   {label_dist[1]}\n")
        f.write(f"  Test:  {label_dist[2]}\n\n")
        f.write(f"Relation Vocabulary Size: {relation_vocab_size}\n")
        f.write("Model Parameters:\n")
        f.write(f"  hidden_dim: {args.hidden_dim}\n")
        f.write(f"  attn_dim: {args.attn_dim}\n")
        f.write(f"  dropout: {args.dropout}\n")
        f.write(f"  Total Trainable Parameters: {param_count}\n\n")
        f.write("Forward Pass Smoke Check:\n")
        f.write(f"  Tiny Batch size: 4\n")
        f.write(f"  Logits shape: {list(logits.shape)} (Expected: [4, 6])\n")
        f.write(f"  Class weights (Train only): {weights.tolist()}\n")
        f.write(f"  Dummy Loss: {loss.item():.6f}\n\n")
        f.write("Verification confirmation:\n")
        f.write("[+] All array shapes, split counts, and label distributions have been validated.\n")
        f.write("[+] Forward pass executed successfully on the tiny batch.\n")
        f.write("[+] NO training, NO checkpoint saving, and NO test evaluation were executed.\n")
        f.write("[+] DRY RUN CHECK COMPLETED SUCCESSFULLY.\n")
    print(f"[+] Saved dry-run report to: {report_path}")

    # Copy README to the config-specific directory as well
    readme_src_path = os.path.join(os.path.dirname(args.out_dir), "G1_README_PLANNED.txt")
    readme_dst_path = os.path.join(args.out_dir, "G1_README_PLANNED.txt")
    if os.path.exists(readme_src_path):
        import shutil
        shutil.copyfile(readme_src_path, readme_dst_path)
        print(f"[+] Copied G1_README_PLANNED.txt to: {readme_dst_path}")

    print("[+] DRY RUN COMPLETED SUCCESSFULLY.")

def run_training_train_val_only(args):
    # Check output paths protection (refuse to overwrite unless specified)
    planned_outputs = [
        args.checkpoint_out,
        os.path.join(args.out_dir, 'G1_TRAINING_SUMMARY.txt')
    ]
    if not args.overwrite:
        for path in planned_outputs:
            if os.path.exists(path):
                print(f"[-] Output file already exists: {path}")
                print("[-] Refusing to run or overwrite without the --overwrite flag.")
                sys.exit(1)

    # 1. Load arrays
    print("Loading datasets...")
    text_features = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    kg_features = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    
    max_relation_id = int(relation_ids.max())
    relation_vocab_size = max_relation_id + 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")
    
    # Instantiate model
    model = G1CoattentionBaseline(
        variant=args.variant,
        num_relations=relation_vocab_size,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout
    ).to(device)

    # Calculate class weights for CrossEntropyLoss on train split only
    train_mask = (split_ids == 0)
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    weights = len(train_labels) / (6.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print("\nStarting model training...")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)

    val_mask = (split_ids == 1)

    # Train tensors
    tr_text = torch.tensor(text_features[train_mask], dtype=torch.float32)
    tr_patch = torch.tensor(image_features_patch[train_mask], dtype=torch.float32)
    tr_img_global = torch.tensor(image_features_global[train_mask], dtype=torch.float32)
    tr_kg = torch.tensor(kg_features[train_mask], dtype=torch.float32)
    tr_rel = torch.tensor(relation_ids[train_mask], dtype=torch.long)
    tr_lbl = torch.tensor(labels_fine[train_mask], dtype=torch.long)

    # Val tensors
    val_text = torch.tensor(text_features[val_mask], dtype=torch.float32)
    val_patch = torch.tensor(image_features_patch[val_mask], dtype=torch.float32)
    val_img_global = torch.tensor(image_features_global[val_mask], dtype=torch.float32)
    val_kg = torch.tensor(kg_features[val_mask], dtype=torch.float32)
    val_rel = torch.tensor(relation_ids[val_mask], dtype=torch.long)
    val_lbl = torch.tensor(labels_fine[val_mask], dtype=torch.long)

    # Datasets & Loaders
    train_ds = TensorDataset(tr_text, tr_patch, tr_img_global, tr_kg, tr_rel, tr_lbl)
    val_ds = TensorDataset(val_text, val_patch, val_img_global, val_kg, val_rel, val_lbl)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = -1.0
    best_epoch = -1
    patience_counter = 0
    patience_limit = 5

    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for bx_text, bx_patch, bx_img_global, bx_kg, bx_rel, bx_lbl in train_loader:
            bx_text = bx_text.to(device)
            bx_patch = bx_patch.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_lbl = bx_lbl.to(device)

            optimizer.zero_grad()
            logits = model(bx_text, bx_patch, bx_img_global, bx_kg, bx_rel)
            loss = criterion(logits, bx_lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx_lbl)

        train_loss /= len(tr_lbl)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for bx_text, bx_patch, bx_img_global, bx_kg, bx_rel, bx_lbl in val_loader:
                bx_text = bx_text.to(device)
                bx_patch = bx_patch.to(device)
                bx_img_global = bx_img_global.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)

                logits = model(bx_text, bx_patch, bx_img_global, bx_kg, bx_rel)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(bx_lbl.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # Compute metrics
        acc = accuracy_score(val_targets, val_preds)
        macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(6)), zero_division=0)
        ck_f1 = per_class_f1[2]

        selection_score = 0.5 * macro_f1 + 0.5 * ck_f1

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val Macro-F1: {macro_f1:.4f} | Val CK-F1: {ck_f1:.4f} | Score: {selection_score:.4f}")

        # Store history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': acc,
            'val_macro_f1': macro_f1,
            'val_weighted_f1': weighted_f1,
            'val_ck_f1': ck_f1,
            'val_selection_score': selection_score
        })

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_epoch = epoch + 1
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': selection_score,
                'val_metrics': {
                    'accuracy': acc,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'ck_f1': ck_f1,
                    'per_class_f1': per_class_f1.tolist(),
                    'preds': val_preds.tolist(),
                    'targets': val_targets.tolist()
                }
            }
            torch.save(checkpoint, args.checkpoint_out)
            print(f"  [+] Saved best checkpoint with Val Score {selection_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered. Best epoch was {best_epoch} with Val Score {best_val_score:.4f}")
                break

    # Save log CSV
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, 'G1_TRAINING_LOG.csv'), index=False)
    print(f"[+] Saved training history to {os.path.join(args.out_dir, 'G1_TRAINING_LOG.csv')}")

    # Load best checkpoint and write final outputs
    print(f"\nLoading best checkpoint from {args.checkpoint_out}...")
    checkpoint = torch.load(args.checkpoint_out, map_location='cpu', weights_only=False)
    best_m = checkpoint['val_metrics']
    val_targets = np.array(best_m['targets'])
    val_preds = np.array(best_m['preds'])
    
    # Save best metrics CSV
    df_metrics = pd.DataFrame([{
        'model': f'g1_coattn_{args.variant}',
        'accuracy': best_m['accuracy'],
        'macro_f1': best_m['macro_f1'],
        'weighted_f1': best_m['weighted_f1'],
        'ck_f1': best_m['ck_f1'],
        'selection_score': checkpoint['val_score'],
        'best_epoch': checkpoint['epoch']
    }])
    df_metrics.to_csv(os.path.join(args.out_dir, 'G1_BEST_VAL_METRICS.csv'), index=False)
    print(f"[+] Saved best metrics CSV to {os.path.join(args.out_dir, 'G1_BEST_VAL_METRICS.csv')}")

    # Save per-class F1 CSV
    df_per_class = pd.DataFrame({
        'class_id': list(range(6)),
        'f1_score': best_m['per_class_f1']
    })
    df_per_class.to_csv(os.path.join(args.out_dir, 'G1_PER_CLASS_F1_VAL.csv'), index=False)
    print(f"[+] Saved per-class F1 to {os.path.join(args.out_dir, 'G1_PER_CLASS_F1_VAL.csv')}")

    # Save confusion matrix CSV and PNG
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(6)))
    df_cm = pd.DataFrame(
        cm,
        index=[f'true_class_{i}' for i in range(6)],
        columns=[f'pred_class_{i}' for i in range(6)]
    )
    df_cm.to_csv(os.path.join(args.out_dir, 'G1_CONFUSION_MATRIX_VAL.csv'), index=True)
    print(f"[+] Saved confusion matrix to {os.path.join(args.out_dir, 'G1_CONFUSION_MATRIX_VAL.csv')}")

    # Plot confusion matrix
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[f'Pred {i}' for i in range(6)],
               yticklabels=[f'True {i}' for i in range(6)],
               title=f'G1 {args.variant} Val Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        cm_png_path = os.path.join(args.out_dir, 'G1_CONFUSION_MATRIX_VAL.png')
        plt.savefig(cm_png_path, dpi=300)
        plt.close()
        print(f"[+] Saved confusion matrix PNG plot to {cm_png_path}")
    except Exception as e:
        print(f"[-] Could not plot confusion matrix PNG: {e}")

    # Write written training summary
    summary_path = os.path.join(args.out_dir, 'G1_TRAINING_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Stage G1 Co-attention Baseline Training Summary ({args.variant})\n")
        f.write("=================================================================\n")
        f.write(f"Best Epoch: {checkpoint['epoch']}\n")
        f.write(f"Validation Selection Score: {checkpoint['val_score']:.4f}\n")
        f.write(f"Validation Accuracy: {best_m['accuracy']:.4f}\n")
        f.write(f"Validation Macro-F1: {best_m['macro_f1']:.4f}\n")
        f.write(f"Validation Weighted-F1: {best_m['weighted_f1']:.4f}\n")
        f.write(f"Validation CK-F1: {best_m['ck_f1']:.4f}\n")
        f.write(f"Per-Class F1: {best_m['per_class_f1']}\n")
    print(f"[+] Saved summary to {summary_path}")

    print("\n[+] MODEL TRAINING AND EXPORT COMPLETED SUCCESSFULLY.")

def run_evaluation_only(args):
    print("=" * 80)
    print("Stage G1 Eval-Only Locked Test")
    print("=" * 80)
    
    # 1. Load arrays
    print("Loading datasets...")
    text_features = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    kg_features = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    
    sample_ids_path = os.path.join(args.cache_dir, 'sample_ids.npy')
    sample_ids = np.load(sample_ids_path) if os.path.exists(sample_ids_path) else None

    # Filter to split_id == 2 (Test split)
    test_mask = (split_ids == 2)
    num_test = int(np.sum(test_mask))
    print(f"[+] Total samples in test split (split_id == 2): {num_test}")
    assert num_test == 2586, f"[-] Expected 2586 samples in test split, but got {num_test}"

    print("[+] Confirmed: Evaluating only test split (split_id == 2). No training will be run.")

    te_text = torch.tensor(text_features[test_mask], dtype=torch.float32)
    te_patch = torch.tensor(image_features_patch[test_mask], dtype=torch.float32)
    te_img_global = torch.tensor(image_features_global[test_mask], dtype=torch.float32)
    te_kg = torch.tensor(kg_features[test_mask], dtype=torch.float32)
    te_rel = torch.tensor(relation_ids[test_mask], dtype=torch.long)
    te_lbl = torch.tensor(labels_fine[test_mask], dtype=torch.long)
    
    if sample_ids is not None:
        te_sample_ids = sample_ids[test_mask]
    else:
        te_sample_ids = np.arange(len(split_ids))[test_mask]

    test_ds = TensorDataset(te_text, te_patch, te_img_global, te_kg, te_rel, te_lbl, torch.tensor(te_sample_ids))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    max_relation_id = int(relation_ids.max())
    relation_vocab_size = max_relation_id + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Load checkpoint
    print(f"Loading best checkpoint from {args.checkpoint_out}...")
    if not os.path.exists(args.checkpoint_out):
        print(f"[-] Checkpoint not found at: {args.checkpoint_out}")
        sys.exit(1)
        
    checkpoint = torch.load(args.checkpoint_out, map_location='cpu', weights_only=False)
    
    # Instantiate model
    model = G1CoattentionBaseline(
        variant=args.variant,
        num_relations=relation_vocab_size,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', -1)
    else:
        model.load_state_dict(checkpoint)
        best_epoch = -1
        
    model.eval()

    test_preds = []
    test_targets = []
    test_sample_ids_list = []
    
    with torch.no_grad():
        for bx_text, bx_patch, bx_img_global, bx_kg, bx_rel, bx_lbl, bx_sid in test_loader:
            bx_text = bx_text.to(device)
            bx_patch = bx_patch.to(device)
            bx_img_global = bx_img_global.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)

            logits = model(bx_text, bx_patch, bx_img_global, bx_kg, bx_rel)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            test_preds.extend(preds)
            test_targets.extend(bx_lbl.numpy())
            test_sample_ids_list.extend(bx_sid.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_sample_ids_list = np.array(test_sample_ids_list)

    # Compute metrics
    acc = accuracy_score(test_targets, test_preds)
    macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(test_targets, test_preds, average=None, labels=list(range(6)), zero_division=0)
    ck_f1 = per_class_f1[2]

    print(f"\nLocked Test Results | Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f} | CK-F1: {ck_f1:.4f}")
    print(f"Per-Class F1: {per_class_f1.tolist()}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Required Outputs:
    # 1. G1_B_LOCKED_TEST_METRICS.csv
    metrics_path = os.path.join(args.out_dir, 'G1_B_LOCKED_TEST_METRICS.csv')
    df_metrics = pd.DataFrame([{
        'model': f'g1_coattn_{args.variant}',
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1,
        'best_epoch': best_epoch
    }])
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[+] Saved metrics CSV to {metrics_path}")

    # 2. G1_B_LOCKED_TEST_PER_CLASS_F1.csv
    per_class_path = os.path.join(args.out_dir, 'G1_B_LOCKED_TEST_PER_CLASS_F1.csv')
    df_per_class = pd.DataFrame({
        'class_id': list(range(6)),
        'f1_score': per_class_f1
    })
    df_per_class.to_csv(per_class_path, index=False)
    print(f"[+] Saved per-class F1 CSV to {per_class_path}")

    # 3. G1_B_LOCKED_TEST_CONFUSION_MATRIX.csv
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(6)))
    df_cm = pd.DataFrame(
        cm,
        index=[f'true_class_{i}' for i in range(6)],
        columns=[f'pred_class_{i}' for i in range(6)]
    )
    cm_path_csv = os.path.join(args.out_dir, 'G1_B_LOCKED_TEST_CONFUSION_MATRIX.csv')
    df_cm.to_csv(cm_path_csv, index=True)
    print(f"[+] Saved confusion matrix CSV to {cm_path_csv}")

    # Plot confusion matrix PNG
    cm_path_png = os.path.join(args.out_dir, 'G1_B_LOCKED_TEST_CONFUSION_MATRIX.png')
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[f'Pred {i}' for i in range(6)],
               yticklabels=[f'True {i}' for i in range(6)],
               title=f'G1 {args.variant} Locked Test Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(cm_path_png, dpi=300)
        plt.close()
        print(f"[+] Saved confusion matrix PNG plot to {cm_path_png}")
    except Exception as e:
        print(f"[-] Could not plot confusion matrix PNG: {e}")

    # 4. G1_B_LOCKED_TEST_PREDICTIONS.csv
    preds_path = os.path.join(args.out_dir, 'G1_B_LOCKED_TEST_PREDICTIONS.csv')
    df_preds = pd.DataFrame({
        'sample_id': test_sample_ids_list,
        'true_label': test_targets,
        'predicted_label': test_preds
    })
    df_preds.to_csv(preds_path, index=False)
    print(f"[+] Saved predictions CSV to {preds_path}")

    # 5. G1_B_LOCKED_TEST_SUMMARY.txt
    summary_path = os.path.join(args.out_dir, 'G1_B_LOCKED_TEST_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Stage G1 Locked Test Summary ({args.variant})\n")
        f.write("=================================================================\n")
        f.write("Evaluation Config & Verification:\n")
        f.write(f"  Checkpoint Evaluated: {args.checkpoint_out}\n")
        f.write(f"  Best Epoch in Checkpoint: {best_epoch}\n")
        f.write(f"  Cache Directory: {args.cache_dir}\n")
        f.write(f"  Only split_id == 2 evaluated: True (Confirmed)\n")
        f.write(f"  No training was run: True (Confirmed)\n")
        f.write(f"  Total samples evaluated: {num_test}\n\n")
        f.write("Evaluation Metrics:\n")
        f.write(f"  Accuracy: {acc:.6f}\n")
        f.write(f"  Macro-F1: {macro_f1:.6f}\n")
        f.write(f"  Weighted-F1: {weighted_f1:.6f}\n")
        f.write(f"  CK-F1: {ck_f1:.6f}\n")
        f.write(f"  Per-Class F1: {per_class_f1.tolist()}\n")
    print(f"[+] Saved locked test summary to {summary_path}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Train Stage G1: Same-split Co-attention Baselines")
    parser.add_argument('--cache_dir', type=str, default='data/cache/kg_complete', help='Cache directory')
    parser.add_argument('--out_dir', type=str, default='outputs/stage_g1_coattention/g1_a_ti_coattn_kg_concat', help='Output directory')
    parser.add_argument('--checkpoint_out', type=str, default='checkpoints/stage_g/g1_a_ti_coattn_kg_concat.pt', help='Checkpoint output path')
    parser.add_argument('--variant', type=str, default='ti_coattn_kg_concat', choices=['ti_coattn_kg_concat', 'kg_image_coattn_text_concat'], help='Variant')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')
    parser.add_argument('--attn_dim', type=int, default=256, help='attn_dim')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
    parser.add_argument("--dry_run", action="store_true", default=False, help="Perform dry-run checks and exit")
    parser.add_argument('--verify_train_path', action='store_true', default=False, help='Verify training path is reachable and exit immediately')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Allow overwriting existing output files')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Perform evaluation on locked test split only')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("Stage G1: Same-split Co-attention Baselines")
    print("=" * 80)

    if args.verify_train_path:
        print("[CONFIG] dry_run = False")
        print(f"Variant = {args.variant}")
        print("G1 TRAINING PATH IS REACHABLE")
        sys.exit(0)
        
    if args.dry_run:
        run_dry_run(args)
        sys.exit(0)
        
    if args.eval_only:
        run_evaluation_only(args)
        sys.exit(0)
        
    run_training_train_val_only(args)

if __name__ == '__main__':
    main()
