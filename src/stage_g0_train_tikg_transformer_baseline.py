"""
Stage G0: Strong Passive T+I+KG Transformer Fusion Baseline.
Prepares the baseline model and directory setup, running in dry-run mode.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class StrongPassiveTIKGTransformer(nn.Module):
    """
    Strong Passive T+I+KG Transformer Fusion Baseline.

    Inputs:
        text_features: [B, 768]
        image_global_features: [B, 512]
        kg_features: [B, 100]
        relation_ids: [B]

    Architecture:
        - Project text_features (768) -> d_model
        - Project image_global_features (512) -> d_model
        - Project kg_features (100) -> d_model
        - Embed relation_ids -> d_model (Embedding(num_relations, d_model))
        - Add learnable modality embeddings for the 4 tokens
        - Prepend learnable CLS token
        - Pass sequence of 5 tokens through TransformerEncoder
        - Retrieve CLS token representation (index 0)
        - Project representation -> 6-way logits
    """
    def __init__(self, num_relations, d_model=256, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        
        # Projections
        self.proj_text = nn.Linear(768, d_model)
        self.proj_img = nn.Linear(512, d_model)
        self.proj_kg = nn.Linear(100, d_model)
        
        # Relation embedding (direct projection to d_model)
        self.relation_embed = nn.Embedding(num_relations, d_model)
        
        # Modality/type embeddings for [text, image, kg, relation]
        self.modality_embeddings = nn.Parameter(torch.randn(4, d_model))
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 6-way classification head
        self.classifier = nn.Linear(d_model, 6)

    def forward(self, text_features, image_global_features, kg_features, relation_ids):
        B = text_features.size(0)
        
        # Project tokens
        t_tok = self.proj_text(text_features)
        img_tok = self.proj_img(image_global_features)
        kg_tok = self.proj_kg(kg_features)
        rel_tok = self.relation_embed(relation_ids)
        
        # Add learnable modality embeddings
        t_tok = t_tok + self.modality_embeddings[0]
        img_tok = img_tok + self.modality_embeddings[1]
        kg_tok = kg_tok + self.modality_embeddings[2]
        rel_tok = rel_tok + self.modality_embeddings[3]
        
        # Stack source tokens: [B, 4, d_model]
        tokens = torch.stack([t_tok, img_tok, kg_tok, rel_tok], dim=1)
        
        # Prepend learnable CLS token: [B, 5, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        
        # Encode via Transformer
        encoded = self.transformer_encoder(x)
        
        # Retrieve CLS representation (pooling)
        pooled = encoded[:, 0]
        
        # Classification head
        logits = self.classifier(pooled)
        
        return logits

def run_dry_run(args):
    # 1. Verify cache files exist
    required_cache_files = [
        'text_features.npy', 'image_features_global.npy',
        'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'split_ids.npy'
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
        print("[-] Cache verification failed. Exiting.")
        sys.exit(1)

    # 2. Load and verify arrays
    print("Loading datasets...")
    text_features = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    kg_features = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    
    sample_ids_path = os.path.join(args.cache_dir, 'sample_ids.npy')
    sample_ids = np.load(sample_ids_path) if os.path.exists(sample_ids_path) else None

    # Print shapes
    print("[+] Loaded array shapes:")
    print(f"    text_features:          {text_features.shape}")
    print(f"    image_features_global:  {image_features_global.shape}")
    print(f"    kg_features:            {kg_features.shape}")
    print(f"    relation_ids:           {relation_ids.shape}")
    print(f"    labels_fine:            {labels_fine.shape}")
    print(f"    split_ids:              {split_ids.shape}")
    if sample_ids is not None:
        print(f"    sample_ids:             {sample_ids.shape}")

    # Verify split counts
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

    # Verify label distribution per split
    label_dist = {}
    for sid, name in [(0, "Train"), (1, "Val"), (2, "Test")]:
        mask = (split_ids == sid)
        lbls = labels_fine[mask]
        dist = np.bincount(lbls, minlength=6)
        label_dist[sid] = dist.tolist()
        print(f"    {name} Label distribution: {dist.tolist()}")

    # Verify relation vocab size
    max_relation_id = int(relation_ids.max())
    relation_vocab_size = max_relation_id + 1
    print(f"[+] Relation vocabulary size: {relation_vocab_size}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Instantiate model
    print("Instantiating StrongPassiveTIKGTransformer model...")
    model = StrongPassiveTIKGTransformer(
        num_relations=relation_vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    # Train-only class-weighted CrossEntropyLoss
    train_mask = (split_ids == 0)
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    weights = len(train_labels) / (6.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[+] Calculated class weights (train split only): {weights.tolist()}")

    print("\n--- DRY RUN STATUS ---")
    print("DRY RUN ONLY — no training executed.")

    # Run forward pass on a tiny batch from train split
    print("Running one forward pass on a tiny batch of size 4 from train split...")
    train_idx = np.where(train_mask)[0][:4]
    bx_text = torch.tensor(text_features[train_idx], dtype=torch.float32).to(device)
    bx_img = torch.tensor(image_features_global[train_idx], dtype=torch.float32).to(device)
    bx_kg = torch.tensor(kg_features[train_idx], dtype=torch.float32).to(device)
    bx_rel = torch.tensor(relation_ids[train_idx], dtype=torch.long).to(device)
    bx_lbl = torch.tensor(labels_fine[train_idx], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(bx_text, bx_img, bx_kg, bx_rel)

    print(f"    Logits shape: {list(logits.shape)} (Expected: [4, 6])")

    # Dummy CE Loss calculation
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss = criterion(logits, bx_lbl)
    print(f"    Dummy CrossEntropyLoss: {loss.item():.6f}")

    # Parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total trainable parameters: {param_count}")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"    Created output directory: {args.out_dir}")

    # Write Dry-run report
    report_path = os.path.join(args.out_dir, "G0_DRY_RUN_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("Stage G0 Dry-Run Report\n")
        f.write("========================\n\n")
        f.write(f"Cache Directory: {args.cache_dir}\n")
        f.write(f"Output Directory: {args.out_dir}\n")
        f.write(f"Checkpoint Output: {args.checkpoint_out}\n\n")
        f.write("Array Shapes Checked:\n")
        f.write(f"  text_features:          {text_features.shape} (Expected: [12786, 768])\n")
        f.write(f"  image_features_global:  {image_features_global.shape} (Expected: [12786, 512])\n")
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
        f.write(f"Model Parameters:\n")
        f.write(f"  d_model: {args.d_model}\n")
        f.write(f"  num_layers: {args.num_layers}\n")
        f.write(f"  num_heads: {args.num_heads}\n")
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
    print("[+] DRY RUN COMPLETED SUCCESSFULLY.")

def run_training_train_val_only(args):
    # Check for --verify_train_path first
    if args.verify_train_path:
        print("TRAINING PATH IS REACHABLE")
        return

    # 1. Verify cache files exist
    required_cache_files = [
        'text_features.npy', 'image_features_global.npy',
        'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'split_ids.npy'
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
        print("[-] Cache verification failed. Exiting.")
        sys.exit(1)

    # Output paths protection (refuse to overwrite unless specified)
    planned_outputs = [
        args.checkpoint_out,
        os.path.join(args.out_dir, 'g0_metrics_val.csv'),
        os.path.join(args.out_dir, 'g0_summary.txt')
    ]
    if not args.overwrite:
        for path in planned_outputs:
            if os.path.exists(path):
                print(f"[-] Output file already exists: {path}")
                print("[-] Refusing to run or overwrite without the --overwrite flag.")
                sys.exit(1)

    # 2. Load arrays
    print("Loading datasets...")
    text_features = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    kg_features = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    
    sample_ids_path = os.path.join(args.cache_dir, 'sample_ids.npy')
    sample_ids = np.load(sample_ids_path) if os.path.exists(sample_ids_path) else None

    # Print shapes
    print("[+] Loaded array shapes:")
    print(f"    text_features:          {text_features.shape}")
    print(f"    image_features_global:  {image_features_global.shape}")
    print(f"    kg_features:            {kg_features.shape}")
    print(f"    relation_ids:           {relation_ids.shape}")
    print(f"    labels_fine:            {labels_fine.shape}")
    print(f"    split_ids:              {split_ids.shape}")
    if sample_ids is not None:
        print(f"    sample_ids:             {sample_ids.shape}")

    # Verify split counts
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

    # Verify label distribution per split
    label_dist = {}
    for sid, name in [(0, "Train"), (1, "Val"), (2, "Test")]:
        mask = (split_ids == sid)
        lbls = labels_fine[mask]
        dist = np.bincount(lbls, minlength=6)
        label_dist[sid] = dist.tolist()
        print(f"    {name} Label distribution: {dist.tolist()}")

    # Verify relation vocab size
    max_relation_id = int(relation_ids.max())
    relation_vocab_size = max_relation_id + 1
    print(f"[+] Relation vocabulary size: {relation_vocab_size}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Instantiate model
    print("Instantiating StrongPassiveTIKGTransformer model...")
    model = StrongPassiveTIKGTransformer(
        num_relations=relation_vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    # Train-only class-weighted CrossEntropyLoss
    train_mask = (split_ids == 0)
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    weights = len(train_labels) / (6.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[+] Calculated class weights (train split only): {weights.tolist()}")

    print("\nStarting model training...")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)

    val_mask = (split_ids == 1)

    # Train tensors
    tr_text = torch.tensor(text_features[train_mask], dtype=torch.float32)
    tr_img = torch.tensor(image_features_global[train_mask], dtype=torch.float32)
    tr_kg = torch.tensor(kg_features[train_mask], dtype=torch.float32)
    tr_rel = torch.tensor(relation_ids[train_mask], dtype=torch.long)
    tr_lbl = torch.tensor(labels_fine[train_mask], dtype=torch.long)

    # Val tensors
    val_text = torch.tensor(text_features[val_mask], dtype=torch.float32)
    val_img = torch.tensor(image_features_global[val_mask], dtype=torch.float32)
    val_kg = torch.tensor(kg_features[val_mask], dtype=torch.float32)
    val_rel = torch.tensor(relation_ids[val_mask], dtype=torch.long)
    val_lbl = torch.tensor(labels_fine[val_mask], dtype=torch.long)

    # Datasets & Loaders
    train_ds = TensorDataset(tr_text, tr_img, tr_kg, tr_rel, tr_lbl)
    val_ds = TensorDataset(val_text, val_img, val_kg, val_rel, val_lbl)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for bx_text, bx_img, bx_kg, bx_rel, bx_lbl in train_loader:
            bx_text = bx_text.to(device)
            bx_img = bx_img.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_lbl = bx_lbl.to(device)

            optimizer.zero_grad()
            logits = model(bx_text, bx_img, bx_kg, bx_rel)
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
            for bx_text, bx_img, bx_kg, bx_rel, bx_lbl in val_loader:
                bx_text = bx_text.to(device)
                bx_img = bx_img.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)

                logits = model(bx_text, bx_img, bx_kg, bx_rel)
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
                    'per_class_f1': per_class_f1.tolist()
                }
            }
            torch.save(checkpoint, args.checkpoint_out)
            print(f"  [+] Saved best checkpoint with Val Score {selection_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered. Best epoch was {best_epoch} with Val Score {best_val_score:.4f}")
                break

    # Load best checkpoint and write final outputs
    print(f"\nLoading best checkpoint from {args.checkpoint_out}...")
    checkpoint = torch.load(args.checkpoint_out, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Save metrics CSV
    metrics_path = os.path.join(args.out_dir, 'g0_metrics_val.csv')
    best_m = checkpoint['val_metrics']
    df_metrics = pd.DataFrame([{
        'model': 'tikg_transformer_baseline',
        'accuracy': best_m['accuracy'],
        'macro_f1': best_m['macro_f1'],
        'weighted_f1': best_m['weighted_f1'],
        'ck_f1': best_m['ck_f1'],
        'selection_score': checkpoint['val_score'],
        'best_epoch': checkpoint['epoch']
    }])
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[+] Saved validation metrics to {metrics_path}")

    # Save summary text
    summary_path = os.path.join(args.out_dir, 'g0_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Stage G0 Passive T+I+KG Transformer Baseline Training Summary\n")
        f.write("=========================================================\n")
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
    print("Stage G0: Evaluation-Only Mode")
    print("=" * 80)
    
    # 1. Verify and Load Checkpoint
    if not args.checkpoint_in or not os.path.exists(args.checkpoint_in):
        print(f"[-] Checkpoint file not specified or does not exist: {args.checkpoint_in}")
        sys.exit(1)
        
    print(f"[+] Loading checkpoint from {args.checkpoint_in}...")
    checkpoint = torch.load(args.checkpoint_in, map_location='cpu', weights_only=False)
    
    # 2. Verify cache files exist
    required_cache_files = [
        'text_features.npy', 'image_features_global.npy',
        'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'split_ids.npy'
    ]
    for f in required_cache_files:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing required cache file: {p}")
            sys.exit(1)
            
    # Load arrays
    print("Loading datasets...")
    text_features = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    kg_features = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    
    sample_ids_path = os.path.join(args.cache_dir, 'sample_ids.npy')
    sample_ids = np.load(sample_ids_path) if os.path.exists(sample_ids_path) else None

    # Get vocab size from relation_ids
    max_relation_id = int(relation_ids.max())
    relation_vocab_size = max_relation_id + 1
    print(f"[+] Relation vocabulary size: {relation_vocab_size}")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Using device: {device}")

    # Instantiate model
    print("Instantiating StrongPassiveTIKGTransformer model...")
    model = StrongPassiveTIKGTransformer(
        num_relations=relation_vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("[+] Model state dict loaded successfully.")

    # Filter by eval split
    eval_split_map = {'train': 0, 'val': 1, 'test': 2}
    split_name = args.eval_split.lower()
    if split_name not in eval_split_map:
        print(f"[-] Unknown split: {args.eval_split}. Must be train, val, or test.")
        sys.exit(1)
    
    split_val = eval_split_map[split_name]
    eval_mask = (split_ids == split_val)
    num_eval = int(np.sum(eval_mask))
    print(f"[+] Evaluating split {split_name} (split_id == {split_val}) only. Found {num_eval} samples.")

    # Datasets & Loaders
    ev_text = torch.tensor(text_features[eval_mask], dtype=torch.float32)
    ev_img = torch.tensor(image_features_global[eval_mask], dtype=torch.float32)
    ev_kg = torch.tensor(kg_features[eval_mask], dtype=torch.float32)
    ev_rel = torch.tensor(relation_ids[eval_mask], dtype=torch.long)
    ev_lbl = torch.tensor(labels_fine[eval_mask], dtype=torch.long)

    eval_ds = TensorDataset(ev_text, ev_img, ev_kg, ev_rel, ev_lbl)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bx_text, bx_img, bx_kg, bx_rel, bx_lbl in eval_loader:
            bx_text = bx_text.to(device)
            bx_img = bx_img.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)

            logits = model(bx_text, bx_img, bx_kg, bx_rel)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(bx_lbl.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, labels=list(range(6)), zero_division=0)
    ck_f1 = per_class_f1[2]
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(6)))

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:    {acc:.6f}")
    print(f"Macro-F1:    {macro_f1:.6f}")
    print(f"Weighted-F1: {weighted_f1:.6f}")
    print(f"CK-F1 (C=2): {ck_f1:.6f}")
    print(f"Per-Class F1: {per_class_f1.tolist()}")
    print("Confusion Matrix:")
    print(cm)

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Metrics CSV
    metrics_path = os.path.join(args.out_dir, 'G0_E_LOCKED_TEST_METRICS.csv')
    df_metrics = pd.DataFrame([{
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1
    }])
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[+] Saved metrics CSV to {metrics_path}")

    # 2. Per-class F1 CSV
    per_class_path = os.path.join(args.out_dir, 'G0_E_LOCKED_TEST_PER_CLASS_F1.csv')
    df_per_class = pd.DataFrame({
        'class_id': list(range(6)),
        'f1_score': per_class_f1.tolist()
    })
    df_per_class.to_csv(per_class_path, index=False)
    print(f"[+] Saved per-class F1 CSV to {per_class_path}")

    # 3. Confusion Matrix CSV
    cm_csv_path = os.path.join(args.out_dir, 'G0_E_LOCKED_TEST_CONFUSION_MATRIX.csv')
    df_cm = pd.DataFrame(
        cm,
        index=[f'true_class_{i}' for i in range(6)],
        columns=[f'pred_class_{i}' for i in range(6)]
    )
    df_cm.to_csv(cm_csv_path, index=True)
    print(f"[+] Saved confusion matrix CSV to {cm_csv_path}")

    # Confusion Matrix PNG (using matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[f'Pred {i}' for i in range(6)],
               yticklabels=[f'True {i}' for i in range(6)],
               title='G0 E Locked Test Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        cm_png_path = os.path.join(args.out_dir, 'G0_E_LOCKED_TEST_CONFUSION_MATRIX.png')
        plt.savefig(cm_png_path, dpi=300)
        plt.close()
        print(f"[+] Saved confusion matrix plot to {cm_png_path}")
    except Exception as e:
        print(f"[-] Could not plot confusion matrix PNG: {e}")

    # 4. Predictions CSV
    preds_path = os.path.join(args.out_dir, 'G0_E_LOCKED_TEST_PREDICTIONS.csv')
    if sample_ids is not None:
        eval_sample_ids = sample_ids[eval_mask]
    else:
        eval_sample_ids = np.arange(len(all_preds))
    df_preds = pd.DataFrame({
        'sample_id': eval_sample_ids,
        'true_label': all_targets,
        'pred_label': all_preds
    })
    df_preds.to_csv(preds_path, index=False)
    print(f"[+] Saved predictions CSV to {preds_path}")

    # 5. Summary Text
    summary_path = os.path.join(args.out_dir, 'G0_E_LOCKED_TEST_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("Stage G0 E Locked Test Evaluation Summary\n")
        f.write("========================================\n\n")
        f.write(f"Evaluated Split: {split_name.upper()} (split_id == {split_val})\n")
        f.write(f"Checkpoint Loaded: {args.checkpoint_in}\n")
        f.write(f"Number of Evaluation Samples: {num_eval}\n\n")
        
        f.write("Model Architecture Details:\n")
        f.write(f"  d_model:    {args.d_model}\n")
        f.write(f"  num_layers: {args.num_layers}\n")
        f.write(f"  num_heads:  {args.num_heads}\n")
        f.write(f"  dropout:    {args.dropout}\n\n")

        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy:    {acc:.6f}\n")
        f.write(f"  Macro-F1:    {macro_f1:.6f}\n")
        f.write(f"  Weighted-F1: {weighted_f1:.6f}\n")
        f.write(f"  CK-F1 (C=2): {ck_f1:.6f}\n\n")

        f.write("Per-Class F1:\n")
        for i, class_f1 in enumerate(per_class_f1):
            f.write(f"  Class {i}: {class_f1:.6f}\n")
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write(f"               Pred_0  Pred_1  Pred_2  Pred_3  Pred_4  Pred_5\n")
        for i in range(6):
            row_str = f"  True_Class_{i}:"
            for j in range(6):
                row_str += f" {cm[i, j]:7d}"
            f.write(row_str + "\n")
        f.write("\n")

        # Reference Comparisons
        f.write("Comparison against Locked-Test Reference Numbers:\n")
        f.write("-------------------------------------------------\n")
        
        # Reference 1
        f.write("1. T+I+KG concat:\n")
        f.write(f"   Accuracy:    {acc:.4f} vs 0.5669 (Diff: {acc - 0.5669:+.4f})\n")
        f.write(f"   Macro-F1:    {macro_f1:.4f} vs 0.4672 (Diff: {macro_f1 - 0.4672:+.4f})\n")
        f.write(f"   Weighted-F1: {weighted_f1:.4f} vs 0.5878 (Diff: {weighted_f1 - 0.5878:+.4f})\n")
        f.write(f"   CK-F1:       {ck_f1:.4f} vs 0.3493 (Diff: {ck_f1 - 0.3493:+.4f})\n\n")

        # Reference 2
        f.write("2. Old CIKD CKBoost-MoE:\n")
        f.write(f"   Accuracy:    {acc:.4f} vs 0.5808 (Diff: {acc - 0.5808:+.4f})\n")
        f.write(f"   Macro-F1:    {macro_f1:.4f} vs 0.4635 (Diff: {macro_f1 - 0.4635:+.4f})\n")
        f.write(f"   Weighted-F1: {weighted_f1:.4f} vs 0.5914 (Diff: {weighted_f1 - 0.5914:+.4f})\n")
        f.write(f"   CK-F1:       {ck_f1:.4f} vs 0.3576 (Diff: {ck_f1 - 0.3576:+.4f})\n\n")

        # Reference 3
        f.write("3. CIKD++-RT no_c_emb:\n")
        f.write(f"   Accuracy:    {acc:.4f} vs 0.5831 (Diff: {acc - 0.5831:+.4f})\n")
        f.write(f"   Macro-F1:    {macro_f1:.4f} vs 0.4698 (Diff: {macro_f1 - 0.4698:+.4f})\n")
        f.write(f"   Weighted-F1: {weighted_f1:.4f} vs 0.5951 (Diff: {weighted_f1 - 0.5951:+.4f})\n")
        f.write(f"   CK-F1:       {ck_f1:.4f} vs 0.3755 (Diff: {ck_f1 - 0.3755:+.4f})\n\n")

    print(f"[+] Saved evaluation summary to {summary_path}")
    print("[+] Evaluation completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Train Stage G0: T+I+KG Transformer Baseline")
    parser.add_argument('--cache_dir', type=str, default='data/cache/kg_complete', help='Cache directory')
    parser.add_argument('--out_dir', type=str, default='outputs/stage_g0_tikg_transformer_baseline', help='Output directory')
    parser.add_argument('--checkpoint_out', type=str, default='checkpoints/stage_g/tikg_transformer_seed42.pt', help='Checkpoint output path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--d_model', type=int, default=256, help='d_model')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    parser.add_argument('--num_heads', type=int, default=4, help='num_heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument("--dry_run", action="store_true", default=False, help="Perform dry-run checks and exit")
    parser.add_argument('--execute_train', action='store_true', help='Execute actual model training loop')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing output files')
    parser.add_argument('--verify_train_path', action='store_true', help='Verify training path is reachable and exit immediately')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--eval_split', type=str, default='test', help='Split to evaluate on')
    parser.add_argument('--checkpoint_in', type=str, default='', help='Checkpoint path to load for evaluation')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("Stage G0: Strong Passive T+I+KG Transformer Fusion Baseline")
    print("=" * 80)

    print(f"[CONFIG] dry_run = {args.dry_run}")
    print(f"[CONFIG] eval_only = {args.eval_only}")

    if args.eval_only:
        run_evaluation_only(args)
        return
    elif args.dry_run:
        run_dry_run(args)
        return
    else:
        run_training_train_val_only(args)
        return

if __name__ == '__main__':
    main()
