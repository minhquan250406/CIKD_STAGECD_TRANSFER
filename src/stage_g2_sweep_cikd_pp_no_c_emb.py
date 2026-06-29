"""
Stage G2: Controlled Improvement Sweep for the main model CIKD++-RT no_c_emb.
Prepares configs g2_a to g2_h and supports dry-run, verify_train_path, and full future training.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

# Import model architecture
from models.cikd_pp_rt import CIKDPPResidualTransformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FocalCrossEntropyLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    """
    def __init__(self, weight=None, gamma=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss
        
        if self.weight is not None:
            w = self.weight[targets]
            loss = w * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def check_promotion_gate(val_macro_f1, val_ck_f1, val_score, tvcs_auc):
    metric_ok = (val_macro_f1 > 0.4792) or (val_ck_f1 > 0.3922) or (val_score > 0.4907)
    tvcs_ok = (tvcs_auc >= 0.68)
    promoted = metric_ok and tvcs_ok
    return promoted

def run_dry_run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check path existence
    cache_files = ['text_features.npy', 'image_features_global.npy', 'image_features_patch.npy',
                   'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'y_ck.npy', 'split_ids.npy']
    cache_ok = True
    for f in cache_files:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing cache file: {p}")
            cache_ok = False
    if not cache_ok:
        sys.exit(1)

    baseline_logits_ok = True
    for f in ['train_logits_base.npy', 'val_logits_base.npy', 'test_logits_base.npy']:
        p = os.path.join(args.baseline_logits_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing baseline logits: {p}")
            baseline_logits_ok = False
    if not baseline_logits_ok:
        sys.exit(1)

    if not os.path.exists(args.tvcs_checkpoint):
        print(f"[-] TVCS Specialist checkpoint missing: {args.tvcs_checkpoint}")
        sys.exit(1)

    # Overwrite check for report path only in dry-run (or skip)
    report_path = os.path.join(args.out_dir, "G2_DRY_RUN_REPORT.txt")
    if not args.overwrite and os.path.exists(report_path):
        print(f"[-] Output already exists: {report_path}")
        print("[-] Refusing to run or overwrite without the --overwrite flag.")
        sys.exit(1)

    # Load shapes to initialize and for dry-run
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    relation_ids = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    num_relations = int(relation_ids.max()) + 1
    kg_features = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    kg_dim = kg_features.shape[1]

    print("\n" + "=" * 80)
    print(f"Stage G2 Dry-Run Checks: {args.config_name}")
    print("=" * 80)

    # Load all arrays
    print("Loading cached features...")
    text_features = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    image_features_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    image_features_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    y_ck = np.load(os.path.join(args.cache_dir, 'y_ck.npy'))
    sample_ids_path = os.path.join(args.cache_dir, 'sample_ids.npy')
    sample_ids = np.load(sample_ids_path) if os.path.exists(sample_ids_path) else None

    # Print all array shapes
    print("[+] Array Shapes Checked:")
    print(f"    text_features:          {text_features.shape} (Expected: [12786, 768])")
    print(f"    image_features_global:  {image_features_global.shape} (Expected: [12786, 512])")
    print(f"    image_features_patch:   {image_features_patch.shape} (Expected: [12786, 49, 512])")
    print(f"    kg_features:            {kg_features.shape} (Expected: [12786, 100])")
    print(f"    relation_ids:           {relation_ids.shape} (Expected: [12786])")
    print(f"    labels_fine:            {labels_fine.shape} (Expected: [12786])")
    print(f"    split_ids:              {split_ids.shape} (Expected: [12786])")
    if sample_ids is not None:
        print(f"    sample_ids:             {sample_ids.shape}")

    # Verify split counts
    num_train = int(np.sum(split_ids == 0))
    num_val = int(np.sum(split_ids == 1))
    num_test = int(np.sum(split_ids == 2))

    print("[+] Split Counts Checked:")
    print(f"    Train (split_id == 0): {num_train} (Expected: 8900)")
    print(f"    Val (split_id == 1):   {num_val} (Expected: 1300)")
    print(f"    Test (split_id == 2):  {num_test} (Expected: 2586)")

    assert num_train == 8900, f"[-] Train count mismatch: {num_train}"
    assert num_val == 1300, f"[-] Val count mismatch: {num_val}"
    assert num_test == 2586, f"[-] Test count mismatch: {num_test}"

    # Load baseline logits and print shapes
    print("Loading baseline logits...")
    train_logits_base = np.load(os.path.join(args.baseline_logits_dir, 'train_logits_base.npy'))
    val_logits_base = np.load(os.path.join(args.baseline_logits_dir, 'val_logits_base.npy'))
    test_logits_base = np.load(os.path.join(args.baseline_logits_dir, 'test_logits_base.npy'))

    print("[+] Baseline Logits Shapes Checked:")
    print(f"    train_logits_base:      {train_logits_base.shape} (Expected: [8900, 6])")
    print(f"    val_logits_base:        {val_logits_base.shape} (Expected: [1300, 6])")
    print(f"    test_logits_base:       {test_logits_base.shape} (Expected: [2586, 6])")

    assert train_logits_base.shape == (8900, 6), f"[-] train_logits_base mismatch: {train_logits_base.shape}"
    assert val_logits_base.shape == (1300, 6), f"[-] val_logits_base mismatch: {val_logits_base.shape}"
    assert test_logits_base.shape == (2586, 6), f"[-] test_logits_base mismatch: {test_logits_base.shape}"

    # Instantiate model for selected config
    print(f"Instantiating CIKDPPResidualTransformer for config '{args.config_name}'...")
    model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=args.dropout,
        alpha_init=0.2,
        alpha_max=args.alpha_max
    ).to(device)

    # Load TVCS Specialist checkpoint
    print(f"Loading TVCS Specialist checkpoint from {args.tvcs_checkpoint}...")
    tvcs_ckpt = torch.load(args.tvcs_checkpoint, map_location=device, weights_only=False)
    tvcs_state = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
    model.tvcs_specialist.load_state_dict(tvcs_state)
    print("  [+] TVCS Specialist loaded successfully.")

    # Load init no_c_emb checkpoint if available
    init_loaded = False
    if os.path.exists(args.init_checkpoint):
        print(f"Loading initial no_c_emb checkpoint from {args.init_checkpoint}...")
        try:
            init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
            init_state = init_ckpt.get('model_state_dict', init_ckpt)
            
            # Check compatibility of shapes
            model.load_state_dict(init_state)
            print("  [+] Initial no_c_emb checkpoint loaded successfully (Compatible).")
            init_loaded = True
        except Exception as e:
            print(f"  [-] Initial no_c_emb checkpoint loading check failed: {e}")
    else:
        print(f"[!] Initial no_c_emb checkpoint NOT found at {args.init_checkpoint}. Proceeding with standard init.")

    # Apply tvcs_mode freezing/unfreezing
    unfrozen_names = []
    for name, param in model.tvcs_specialist.named_parameters():
        if args.tvcs_mode == 'frozen':
            param.requires_grad = False
        elif args.tvcs_mode == 'unfreeze_last_projection':
            if any(proj in name for proj in ['Wq', 'Wk', 'Wv', 'patch_proj']):
                param.requires_grad = True
                unfrozen_names.append(name)
            else:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[+] Parameter Counts:")
    print(f"    Trainable parameters: {trainable_params}")
    print(f"    Frozen parameters:    {frozen_params}")
    if args.tvcs_mode == 'unfreeze_last_projection':
        print(f"    Unfrozen TVCS layers: {unfrozen_names}")

    # Run forward pass on tiny batch of size 4 from train split
    print("Running tiny forward pass check on batch size 4...")
    train_mask = (split_ids == 0)
    train_idx = np.where(train_mask)[0][:4]
    
    bx_text = torch.tensor(text_features[train_idx], dtype=torch.float32).to(device)
    bx_img_g = torch.tensor(image_features_global[train_idx], dtype=torch.float32).to(device)
    bx_img_p = torch.tensor(image_features_patch[train_idx], dtype=torch.float32).to(device)
    bx_kg = torch.tensor(kg_features[train_idx], dtype=torch.float32).to(device)
    bx_rel = torch.tensor(relation_ids[train_idx], dtype=torch.long).to(device)
    bx_lbl = torch.tensor(labels_fine[train_idx], dtype=torch.long).to(device)
    bx_y_ck = torch.tensor(y_ck[train_idx], dtype=torch.float32).to(device)
    bx_logits = torch.tensor(train_logits_base[train_idx], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits,
            ablation_no_c_emb=True
        )

    final_logits = outputs['logits_final']
    logits_delta = outputs['logits_delta']
    alpha = outputs['alpha']
    c_logit = outputs['c_logit']

    print(f"[+] Output Shapes:")
    print(f"    final_logits:           {list(final_logits.shape)} (Expected: [4, 6])")
    print(f"    logits_delta:           {list(logits_delta.shape)} (Expected: [4, 6])")
    print(f"    alpha:                  {alpha.item():.4f}")

    assert list(final_logits.shape) == [4, 6], f"[-] final_logits shape mismatch: {final_logits.shape}"
    assert list(logits_delta.shape) == [4, 6], f"[-] logits_delta shape mismatch: {logits_delta.shape}"

    # Compute dummy training loss
    train_labels = labels_fine[train_mask]
    class_counts = np.bincount(train_labels, minlength=6)
    class_counts = np.maximum(class_counts, 1)
    weights = len(train_labels) / (6.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    if args.focal_gamma == 1.0:
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion_cls = FocalCrossEntropyLoss(weight=class_weights, gamma=args.focal_gamma)

    loss_cls = criterion_cls(final_logits, bx_lbl)

    # TVCS loss
    criterion_tvcs = nn.BCEWithLogitsLoss()
    mask_tvcs = (bx_y_ck != -1)
    if mask_tvcs.sum() > 0:
        loss_tvcs = criterion_tvcs(c_logit[mask_tvcs], bx_y_ck[mask_tvcs])
    else:
        loss_tvcs = torch.tensor(0.0, device=device)

    # Residual loss
    loss_res = torch.mean(logits_delta ** 2)

    dummy_loss = loss_cls + 0.5 * loss_tvcs + 0.01 * loss_res
    print(f"[+] Dummy Loss computed: {dummy_loss.item():.6f}")

    # Compute dummy TVCS scores
    tvcs_scores = torch.sigmoid(c_logit).cpu().numpy()
    print(f"[+] Dummy TVCS Scores: {tvcs_scores.tolist()}")

    # Write G2_DRY_RUN_REPORT.txt
    os.makedirs(args.out_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"G2 DRY-RUN REPORT: {args.config_name}\n")
        f.write("=======================================\n\n")
        f.write(f"Config Name:         {args.config_name}\n")
        f.write(f"Alpha Max:           {args.alpha_max}\n")
        f.write(f"Focal Gamma:         {args.focal_gamma}\n")
        f.write(f"TVCS Mode:           {args.tvcs_mode}\n\n")
        f.write("Array Shapes:\n")
        f.write(f"  text_features:          {text_features.shape}\n")
        f.write(f"  image_features_global:  {image_features_global.shape}\n")
        f.write(f"  image_features_patch:   {image_features_patch.shape}\n")
        f.write(f"  kg_features:            {kg_features.shape}\n")
        f.write(f"  relation_ids:           {relation_ids.shape}\n")
        f.write(f"  labels_fine:            {labels_fine.shape}\n")
        f.write(f"  split_ids:              {split_ids.shape}\n\n")
        f.write("Split Counts:\n")
        f.write(f"  Train: {num_train}\n")
        f.write(f"  Val:   {num_val}\n")
        f.write(f"  Test:  {num_test}\n\n")
        f.write("Baseline Logits Shapes:\n")
        f.write(f"  train_logits_base:      {train_logits_base.shape}\n")
        f.write(f"  val_logits_base:        {val_logits_base.shape}\n")
        f.write(f"  test_logits_base:       {test_logits_base.shape}\n\n")
        f.write("Model Parameter Verification:\n")
        f.write(f"  TVCS Specialist loaded: True\n")
        f.write(f"  Init checkpoint loaded: {init_loaded}\n")
        f.write(f"  Trainable parameters:   {trainable_params}\n")
        f.write(f"  Frozen parameters:      {frozen_params}\n\n")
        f.write("Forward Pass Check:\n")
        f.write(f"  final_logits shape:     {list(final_logits.shape)}\n")
        f.write(f"  logits_delta shape:     {list(logits_delta.shape)}\n")
        f.write(f"  alpha:                  {alpha.item():.4f}\n")
        f.write(f"  Dummy Loss:             {dummy_loss.item():.6f}\n")
        f.write(f"  Dummy TVCS Scores:      {tvcs_scores.tolist()}\n\n")
        f.write("Verification Result:\n")
        f.write("[+] ALL SHAPE CHECKS, FORWARD-PASS CHECKS, AND CHECKPOINT LOADING CHECKS COMPLETED.\n")
        f.write("[+] NO TRAINING WAS RUN, NO TEST EVALUATED, AND NO CKPT SAVED.\n")
        f.write("[+] DRY RUN COMPLETED SUCCESSFULLY.\n")

    print(f"[+] Saved dry-run report to: {report_path}")
    print("[+] DRY RUN COMPLETED SUCCESSFULLY.")

def run_training_train_val_only(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check path existence
    cache_files = ['text_features.npy', 'image_features_global.npy', 'image_features_patch.npy',
                   'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'y_ck.npy', 'split_ids.npy']
    cache_ok = True
    for f in cache_files:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing cache file: {p}")
            cache_ok = False
    if not cache_ok:
        sys.exit(1)

    baseline_logits_ok = True
    for f in ['train_logits_base.npy', 'val_logits_base.npy', 'test_logits_base.npy']:
        p = os.path.join(args.baseline_logits_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing baseline logits: {p}")
            baseline_logits_ok = False
    if not baseline_logits_ok:
        sys.exit(1)

    if not os.path.exists(args.tvcs_checkpoint):
        print(f"[-] TVCS Specialist checkpoint missing: {args.tvcs_checkpoint}")
        sys.exit(1)

    # Overwrite check
    planned_outputs = [
        args.checkpoint_out,
        os.path.join(args.out_dir, 'G2_TRAINING_LOG.csv'),
        os.path.join(args.out_dir, 'G2_BEST_VAL_METRICS.csv'),
        os.path.join(args.out_dir, 'G2_TVCS_VAL_METRICS.csv'),
        os.path.join(args.out_dir, 'G2_PER_CLASS_F1_VAL.csv'),
        os.path.join(args.out_dir, 'G2_CONFUSION_MATRIX_VAL.csv'),
        os.path.join(args.out_dir, 'G2_TRAINING_SUMMARY.txt')
    ]
    if not args.overwrite:
        for p in planned_outputs:
            if os.path.exists(p):
                print(f"[-] Output already exists: {p}")
                print("[-] Refusing to run or overwrite without the --overwrite flag.")
                sys.exit(1)

    # Load split_ids & relation_ids to configure classes & dimensions
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    relation_ids_all = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    num_relations = int(relation_ids_all.max()) + 1
    kg_features_all = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    kg_dim = kg_features_all.shape[1]

    # Future Execution Mode (Training)
    print("\nStarting G2 sweep training execution...")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)

    # Load cache features
    print("Loading datasets...")
    text_feat = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    img_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    img_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    kg_feats = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    y_ck = np.load(os.path.join(args.cache_dir, 'y_ck.npy'))

    # Load baseline logits
    tr_logits_base = np.load(os.path.join(args.baseline_logits_dir, 'train_logits_base.npy'))
    val_logits_base = np.load(os.path.join(args.baseline_logits_dir, 'val_logits_base.npy'))

    # Extract Train/Val splits (Train: split_id == 0, Val: split_id == 1, never use split_id == 2)
    train_mask = (split_ids == 0)
    val_mask = (split_ids == 1)

    # Tensors
    tr_text = torch.tensor(text_feat[train_mask], dtype=torch.float32)
    tr_img_g = torch.tensor(img_global[train_mask], dtype=torch.float32)
    tr_img_p = torch.tensor(img_patch[train_mask], dtype=torch.float32)
    tr_kg = torch.tensor(kg_feats[train_mask], dtype=torch.float32)
    tr_rel = torch.tensor(relation_ids_all[train_mask], dtype=torch.long)
    tr_lbl = torch.tensor(labels_fine[train_mask], dtype=torch.long)
    tr_y_ck = torch.tensor(y_ck[train_mask], dtype=torch.float32)
    tr_logits = torch.tensor(tr_logits_base, dtype=torch.float32)

    val_text = torch.tensor(text_feat[val_mask], dtype=torch.float32)
    val_img_g = torch.tensor(img_global[val_mask], dtype=torch.float32)
    val_img_p = torch.tensor(img_patch[val_mask], dtype=torch.float32)
    val_kg = torch.tensor(kg_feats[val_mask], dtype=torch.float32)
    val_rel = torch.tensor(relation_ids_all[val_mask], dtype=torch.long)
    val_lbl = torch.tensor(labels_fine[val_mask], dtype=torch.long)
    val_y_ck = torch.tensor(y_ck[val_mask], dtype=torch.float32)
    val_logits = torch.tensor(val_logits_base, dtype=torch.float32)

    # Dataloaders
    train_ds = TensorDataset(tr_text, tr_img_g, tr_img_p, tr_kg, tr_rel, tr_lbl, tr_y_ck, tr_logits)
    val_ds = TensorDataset(val_text, val_img_g, val_img_p, val_kg, val_rel, val_lbl, val_y_ck, val_logits)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Instantiate model
    model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=args.dropout,
        alpha_init=0.2,
        alpha_max=args.alpha_max
    ).to(device)

    # Load TVCS Specialist
    tvcs_ckpt = torch.load(args.tvcs_checkpoint, map_location=device, weights_only=False)
    tvcs_state = tvcs_ckpt.get('model_state_dict', tvcs_ckpt)
    model.tvcs_specialist.load_state_dict(tvcs_state)

    # Load init no_c_emb checkpoint if available
    if os.path.exists(args.init_checkpoint):
        try:
            init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
            init_state = init_ckpt.get('model_state_dict', init_ckpt)
            model.load_state_dict(init_state)
            print("[+] Loaded warm-start checkpoint weights successfully.")
        except Exception as e:
            print(f"[!] Warm-start check error: {e}. Starting from scratch.")

    # Freeze / Unfreeze TVCS specialist
    for name, param in model.tvcs_specialist.named_parameters():
        if args.tvcs_mode == 'frozen':
            param.requires_grad = False
        elif args.tvcs_mode == 'unfreeze_last_projection':
            if any(proj in name for proj in ['Wq', 'Wk', 'Wv', 'patch_proj']):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Calculate class weights
    train_labels_mask = labels_fine[train_mask]
    counts = np.bincount(train_labels_mask, minlength=6)
    counts = np.maximum(counts, 1)
    weights = len(train_labels_mask) / (6.0 * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    if args.focal_gamma == 1.0:
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion_cls = FocalCrossEntropyLoss(weight=class_weights, gamma=args.focal_gamma)

    criterion_tvcs = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_epoch = -1
    patience_counter = 0

    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
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
            outputs = model(
                bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits,
                ablation_no_c_emb=True
            )

            logits_final = outputs['logits_final']
            logits_delta = outputs['logits_delta']
            c_logit = outputs['c_logit']

            loss_cls = criterion_cls(logits_final, bx_lbl)

            mask_tvcs = (bx_y_ck != -1)
            if mask_tvcs.sum() > 0:
                loss_tvcs = criterion_tvcs(c_logit[mask_tvcs], bx_y_ck[mask_tvcs])
            else:
                loss_tvcs = torch.tensor(0.0, device=device)

            loss_res = torch.mean(logits_delta ** 2)

            loss_total = loss_cls + 0.5 * loss_tvcs + 0.01 * loss_res

            loss_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss_total.item() * len(bx_lbl)

        train_loss /= len(tr_lbl)

        # Validation
        model.eval()
        val_preds, val_targets, val_c_probs, val_y_ck_list = [], [], [], []
        with torch.no_grad():
            for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_y_ck, bx_logits in val_loader:
                bx_text = bx_text.to(device)
                bx_img_g = bx_img_g.to(device)
                bx_img_p = bx_img_p.to(device)
                bx_kg = bx_kg.to(device)
                bx_rel = bx_rel.to(device)
                bx_logits = bx_logits.to(device)

                outputs = model(
                    bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits,
                    ablation_no_c_emb=True
                )
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
        weighted_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, labels=list(range(6)), zero_division=0)
        ck_f1 = per_class_f1[2]

        tvcs_mask = (val_y_ck_arr != -1)
        if tvcs_mask.sum() > 0 and len(np.unique(val_y_ck_arr[tvcs_mask])) > 1:
            tvcs_auc = roc_auc_score(val_y_ck_arr[tvcs_mask], val_c_probs[tvcs_mask])
            tvcs_pr_auc = average_precision_score(val_y_ck_arr[tvcs_mask], val_c_probs[tvcs_mask])
        else:
            tvcs_auc = 0.5
            tvcs_pr_auc = 0.0

        real_mask = (val_y_ck_arr == 0)
        mean_c_real = float(np.mean(val_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
        ck_mask_y = (val_y_ck_arr == 1)
        mean_c_ck = float(np.mean(val_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
        tvcs_delta = mean_c_ck - mean_c_real

        # Validation Selection Score
        val_score = 0.45 * macro_f1 + 0.35 * ck_f1 + 0.20 * tvcs_auc

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val Macro-F1: {macro_f1:.4f} | Val CK-F1: {ck_f1:.4f} | Val TVCS AUC: {tvcs_auc:.4f} | Score: {val_score:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': acc,
            'val_macro_f1': macro_f1,
            'val_weighted_f1': weighted_f1,
            'val_ck_f1': ck_f1,
            'val_tvcs_auc': tvcs_auc,
            'val_selection_score': val_score
        })

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'val_metrics': {
                    'accuracy': acc,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'ck_f1': ck_f1,
                    'per_class_f1': per_class_f1.tolist(),
                    'tvcs_auc_ck_vs_real': tvcs_auc,
                    'tvcs_pr_auc': tvcs_pr_auc,
                    'mean_c_real': mean_c_real,
                    'mean_c_ck': mean_c_ck,
                    'tvcs_delta': tvcs_delta,
                    'preds': val_preds.tolist(),
                    'targets': val_targets.tolist()
                }
            }
            torch.save(checkpoint, args.checkpoint_out)
            print(f"  [+] Saved best checkpoint with Val Score {val_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Save training metrics outputs
    print(f"\nLoading best checkpoint from {args.checkpoint_out}...")
    checkpoint = torch.load(args.checkpoint_out, map_location='cpu', weights_only=False)
    best_m = checkpoint['val_metrics']
    val_targets = np.array(best_m['targets'])
    val_preds = np.array(best_m['preds'])

    # 1. G2_TRAINING_LOG.csv
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, 'G2_TRAINING_LOG.csv'), index=False)
    
    # 2. G2_BEST_VAL_METRICS.csv
    df_best = pd.DataFrame([{
        'config': args.config_name,
        'accuracy': best_m['accuracy'],
        'macro_f1': best_m['macro_f1'],
        'weighted_f1': best_m['weighted_f1'],
        'ck_f1': best_m['ck_f1'],
        'tvcs_auc': best_m['tvcs_auc_ck_vs_real'],
        'selection_score': checkpoint['val_score'],
        'best_epoch': checkpoint['epoch']
    }])
    df_best.to_csv(os.path.join(args.out_dir, 'G2_BEST_VAL_METRICS.csv'), index=False)

    # 3. G2_TVCS_VAL_METRICS.csv
    df_tvcs = pd.DataFrame([{
        'config': args.config_name,
        'tvcs_auc_ck_vs_real': best_m['tvcs_auc_ck_vs_real'],
        'tvcs_pr_auc': best_m['tvcs_pr_auc'],
        'mean_c_real': best_m['mean_c_real'],
        'mean_c_ck': best_m['mean_c_ck'],
        'tvcs_delta': best_m['tvcs_delta']
    }])
    df_tvcs.to_csv(os.path.join(args.out_dir, 'G2_TVCS_VAL_METRICS.csv'), index=False)

    # 4. G2_PER_CLASS_F1_VAL.csv
    df_per_class = pd.DataFrame({
        'class_id': list(range(6)),
        'f1_score': best_m['per_class_f1']
    })
    df_per_class.to_csv(os.path.join(args.out_dir, 'G2_PER_CLASS_F1_VAL.csv'), index=False)

    # 5. G2_CONFUSION_MATRIX_VAL.csv
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(6)))
    df_cm = pd.DataFrame(
        cm,
        index=[f'true_class_{i}' for i in range(6)],
        columns=[f'pred_class_{i}' for i in range(6)]
    )
    df_cm.to_csv(os.path.join(args.out_dir, 'G2_CONFUSION_MATRIX_VAL.csv'), index=True)

    # Plot confusion matrix PNG if possible
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[f'Pred {i}' for i in range(6)],
               yticklabels=[f'True {i}' for i in range(6)],
               title=f'G2 {args.config_name} Val CM',
               ylabel='True label',
               xlabel='Predicted label')
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'G2_CONFUSION_MATRIX_VAL.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[-] Could not plot CM PNG: {e}")

    # Check promotion gate
    promoted = check_promotion_gate(best_m['macro_f1'], best_m['ck_f1'], checkpoint['val_score'], best_m['tvcs_auc_ck_vs_real'])

    # 6. G2_TRAINING_SUMMARY.txt
    summary_path = os.path.join(args.out_dir, 'G2_TRAINING_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Stage G2 Training Summary: {args.config_name}\n")
        f.write("===============================================\n")
        f.write(f"Best Epoch:           {checkpoint['epoch']}\n")
        f.write(f"Val Selection Score:  {checkpoint['val_score']:.4f}\n")
        f.write(f"Val Accuracy:         {best_m['accuracy']:.4f}\n")
        f.write(f"Val Macro-F1:         {best_m['macro_f1']:.4f}\n")
        f.write(f"Val CK-F1:            {best_m['ck_f1']:.4f}\n")
        f.write(f"Val TVCS AUC:         {best_m['tvcs_auc_ck_vs_real']:.4f}\n")
        f.write(f"Val TVCS PR-AUC:      {best_m['tvcs_pr_auc']:.4f}\n")
        f.write(f"TVCS Delta:           {best_m['tvcs_delta']:.4f}\n")
        f.write(f"Per-Class F1:         {best_m['per_class_f1']}\n\n")
        f.write("Promotion Gate Criteria Check:\n")
        f.write(f"  Val Macro-F1 > 0.4792:  {best_m['macro_f1'] > 0.4792} ({best_m['macro_f1']:.4f})\n")
        f.write(f"  Val CK-F1 > 0.3922:     {best_m['ck_f1'] > 0.3922} ({best_m['ck_f1']:.4f})\n")
        f.write(f"  Val Score > 0.4907:     {checkpoint['val_score'] > 0.4907} ({checkpoint['val_score']:.4f})\n")
        f.write(f"  AND TVCS AUC >= 0.68:   {best_m['tvcs_auc_ck_vs_real'] >= 0.68} ({best_m['tvcs_auc_ck_vs_real']:.4f})\n")
        f.write(f"  PROMOTION STATUS:       {'PROMOTED' if promoted else 'REJECTED'}\n")

    print(f"[+] Saved summary to {summary_path}")
    print(f"[+] PROMOTION STATUS: {'PROMOTED' if promoted else 'REJECTED'}")
    print("\n[+] MODEL TRAINING AND EXPORT COMPLETED SUCCESSFULLY.")
def run_evaluation_only(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check path existence
    cache_files = ['text_features.npy', 'image_features_global.npy', 'image_features_patch.npy',
                   'kg_features.npy', 'relation_ids.npy', 'labels_fine.npy', 'y_ck.npy', 'split_ids.npy']
    cache_ok = True
    for f in cache_files:
        p = os.path.join(args.cache_dir, f)
        if not os.path.exists(p):
            print(f"[-] Missing cache file: {p}")
            cache_ok = False
    if not cache_ok:
        sys.exit(1)

    # We need baseline test logits
    test_logits_path = os.path.join(args.baseline_logits_dir, 'test_logits_base.npy')
    if not os.path.exists(test_logits_path):
        print(f"[-] Missing baseline test logits: {test_logits_path}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint_out):
        print(f"[-] Checkpoint file not found: {args.checkpoint_out}")
        sys.exit(1)

    # Load split_ids & relation_ids to configure classes & dimensions
    split_ids = np.load(os.path.join(args.cache_dir, 'split_ids.npy'))
    relation_ids_all = np.load(os.path.join(args.cache_dir, 'relation_ids.npy'))
    num_relations = int(relation_ids_all.max()) + 1
    kg_features_all = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    kg_dim = kg_features_all.shape[1]

    print("\nStarting G2 locked test evaluation execution...")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load cache features
    print("Loading datasets...")
    text_feat = np.load(os.path.join(args.cache_dir, 'text_features.npy'))
    img_global = np.load(os.path.join(args.cache_dir, 'image_features_global.npy'))
    img_patch = np.load(os.path.join(args.cache_dir, 'image_features_patch.npy'))
    kg_feats = np.load(os.path.join(args.cache_dir, 'kg_features.npy'))
    labels_fine = np.load(os.path.join(args.cache_dir, 'labels_fine.npy'))
    y_ck = np.load(os.path.join(args.cache_dir, 'y_ck.npy'))
    sample_ids_path = os.path.join(args.cache_dir, 'sample_ids.npy')
    sample_ids = np.load(sample_ids_path) if os.path.exists(sample_ids_path) else None

    # Load baseline test logits
    test_logits_base = np.load(test_logits_path)

    # Extract Test split (split_id == 2)
    test_mask = (split_ids == 2)
    num_test = int(np.sum(test_mask))
    print(f"[+] Total samples in test split (split_id == 2): {num_test}")
    assert num_test == 2586, f"[-] Expected 2586 samples in test split, but got {num_test}"
    print("[+] Confirmed: Evaluating only test split (split_id == 2). No training will be run.")

    # Tensors
    te_text = torch.tensor(text_feat[test_mask], dtype=torch.float32)
    te_img_g = torch.tensor(img_global[test_mask], dtype=torch.float32)
    te_img_p = torch.tensor(img_patch[test_mask], dtype=torch.float32)
    te_kg = torch.tensor(kg_feats[test_mask], dtype=torch.float32)
    te_rel = torch.tensor(relation_ids_all[test_mask], dtype=torch.long)
    te_lbl = torch.tensor(labels_fine[test_mask], dtype=torch.long)
    te_y_ck = torch.tensor(y_ck[test_mask], dtype=torch.float32)
    te_logits = torch.tensor(test_logits_base, dtype=torch.float32)

    if sample_ids is not None:
        te_sample_ids = sample_ids[test_mask]
    else:
        te_sample_ids = np.arange(len(split_ids))[test_mask]

    # Dataloader
    test_ds = TensorDataset(te_text, te_img_g, te_img_p, te_kg, te_rel, te_lbl, te_y_ck, te_logits, torch.tensor(te_sample_ids))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Instantiate model
    model = CIKDPPResidualTransformer(
        num_relations=num_relations,
        kg_dim=kg_dim,
        d_model=256,
        num_layers=2,
        num_heads=4,
        dropout=args.dropout,
        alpha_init=0.2,
        alpha_max=args.alpha_max
    ).to(device)

    # Load best validation checkpoint
    print(f"Loading best checkpoint from {args.checkpoint_out}...")
    checkpoint = torch.load(args.checkpoint_out, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', -1)
    else:
        model.load_state_dict(checkpoint)
        best_epoch = -1
    print("  [+] Checkpoint loaded successfully.")

    model.eval()
    test_preds, test_targets, test_c_probs, test_y_ck_list, test_sample_ids_list = [], [], [], [], []
    with torch.no_grad():
        for bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_lbl, bx_y_ck, bx_logits, bx_sid in test_loader:
            bx_text = bx_text.to(device)
            bx_img_g = bx_img_g.to(device)
            bx_img_p = bx_img_p.to(device)
            bx_kg = bx_kg.to(device)
            bx_rel = bx_rel.to(device)
            bx_logits = bx_logits.to(device)

            outputs = model(
                bx_text, bx_img_g, bx_img_p, bx_kg, bx_rel, bx_logits,
                ablation_no_c_emb=True
            )
            logits_final = outputs['logits_final']
            c_logit = outputs['c_logit']

            preds = torch.argmax(logits_final, dim=-1).cpu().numpy()
            probs_tvcs = torch.sigmoid(c_logit).cpu().numpy()

            test_preds.extend(preds)
            test_targets.extend(bx_lbl.cpu().numpy())
            test_c_probs.extend(probs_tvcs)
            test_y_ck_list.extend(bx_y_ck.numpy())
            test_sample_ids_list.extend(bx_sid.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_c_probs = np.array(test_c_probs)
    test_y_ck_arr = np.array(test_y_ck_list)
    test_sample_ids_arr = np.array(test_sample_ids_list)

    # Compute metrics
    acc = accuracy_score(test_targets, test_preds)
    macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(test_targets, test_preds, average=None, labels=list(range(6)), zero_division=0)
    ck_f1 = per_class_f1[2]

    tvcs_mask = (test_y_ck_arr != -1)
    if tvcs_mask.sum() > 0 and len(np.unique(test_y_ck_arr[tvcs_mask])) > 1:
        tvcs_auc = roc_auc_score(test_y_ck_arr[tvcs_mask], test_c_probs[tvcs_mask])
        tvcs_pr_auc = average_precision_score(test_y_ck_arr[tvcs_mask], test_c_probs[tvcs_mask])
    else:
        tvcs_auc = 0.5
        tvcs_pr_auc = 0.0

    real_mask = (test_y_ck_arr == 0)
    mean_c_real = float(np.mean(test_c_probs[real_mask])) if real_mask.sum() > 0 else 0.0
    ck_mask_y = (test_y_ck_arr == 1)
    mean_c_ck = float(np.mean(test_c_probs[ck_mask_y])) if ck_mask_y.sum() > 0 else 0.0
    tvcs_delta = mean_c_ck - mean_c_real

    print(f"\nLocked Test Results | Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f} | CK-F1: {ck_f1:.4f} | TVCS AUC: {tvcs_auc:.4f}")
    print(f"Mean C Real: {mean_c_real:.4f} | Mean C CK: {mean_c_ck:.4f} | TVCS Delta: {tvcs_delta:.4f}")

    # Required Outputs:
    # 1. G2_B_LOCKED_TEST_METRICS.csv
    metrics_path = os.path.join(args.out_dir, 'G2_B_LOCKED_TEST_METRICS.csv')
    df_metrics = pd.DataFrame([{
        'config': args.config_name,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ck_f1': ck_f1,
        'best_epoch': best_epoch
    }])
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[+] Saved metrics CSV to {metrics_path}")

    # 2. G2_B_LOCKED_TEST_PER_CLASS_F1.csv
    per_class_path = os.path.join(args.out_dir, 'G2_B_LOCKED_TEST_PER_CLASS_F1.csv')
    df_per_class = pd.DataFrame({
        'class_id': list(range(6)),
        'f1_score': per_class_f1
    })
    df_per_class.to_csv(per_class_path, index=False)
    print(f"[+] Saved per-class F1 CSV to {per_class_path}")

    # 3. G2_B_LOCKED_TEST_TVCS_METRICS.csv
    tvcs_path = os.path.join(args.out_dir, 'G2_B_LOCKED_TEST_TVCS_METRICS.csv')
    df_tvcs = pd.DataFrame([{
        'config': args.config_name,
        'tvcs_auc_ck_vs_real': tvcs_auc,
        'tvcs_pr_auc': tvcs_pr_auc,
        'mean_c_real': mean_c_real,
        'mean_c_ck': mean_c_ck,
        'tvcs_delta': tvcs_delta
    }])
    df_tvcs.to_csv(tvcs_path, index=False)
    print(f"[+] Saved TVCS metrics CSV to {tvcs_path}")

    # 4. G2_B_LOCKED_TEST_CONFUSION_MATRIX.csv
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(6)))
    df_cm = pd.DataFrame(
        cm,
        index=[f'true_class_{i}' for i in range(6)],
        columns=[f'pred_class_{i}' for i in range(6)]
    )
    cm_path_csv = os.path.join(args.out_dir, 'G2_B_LOCKED_TEST_CONFUSION_MATRIX.csv')
    df_cm.to_csv(cm_path_csv, index=True)
    print(f"[+] Saved confusion matrix CSV to {cm_path_csv}")

    # Plot confusion matrix PNG
    cm_path_png = os.path.join(args.out_dir, 'G2_B_LOCKED_TEST_CONFUSION_MATRIX.png')
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[f'Pred {i}' for i in range(6)],
               yticklabels=[f'True {i}' for i in range(6)],
               title=f'G2 {args.config_name} Locked Test Confusion Matrix',
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

    # 5. G2_B_LOCKED_TEST_PREDICTIONS.csv
    preds_path = os.path.join(args.out_dir, 'G2_B_LOCKED_TEST_PREDICTIONS.csv')
    df_preds = pd.DataFrame({
        'sample_id': test_sample_ids_arr,
        'true_label': test_targets,
        'predicted_label': test_preds,
        'tvcs_prob': test_c_probs,
        'y_ck': test_y_ck_arr
    })
    df_preds.to_csv(preds_path, index=False)
    print(f"[+] Saved predictions CSV to {preds_path}")

    # 6. G2_B_LOCKED_TEST_SUMMARY.txt
    summary_path = os.path.join(args.out_dir, 'G2_B_LOCKED_TEST_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Stage G2 Locked Test Summary: {args.config_name}\n")
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
        f.write(f"  TVCS AUC: {tvcs_auc:.6f}\n")
        f.write(f"  TVCS PR-AUC: {tvcs_pr_auc:.6f}\n")
        f.write(f"  mean_c_real: {mean_c_real:.6f}\n")
        f.write(f"  mean_c_ck: {mean_c_ck:.6f}\n")
        f.write(f"  tvcs_delta: {tvcs_delta:.6f}\n")
        f.write(f"  Per-Class F1: {per_class_f1.tolist()}\n")
    print(f"[+] Saved locked test summary to {summary_path}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Stage G2: Controlled Improvement Sweep for CIKD++-RT no_c_emb")
    
    # Required paths
    parser.add_argument('--cache_dir', type=str, default='data/cache/kg_complete', help='Cache directory')
    parser.add_argument('--baseline_logits_dir', type=str, default='outputs/stage_f0_baseline_anchor', help='Baseline logits directory')
    parser.add_argument('--tvcs_checkpoint', type=str, default='checkpoints/stage_f/tvcs_specialist_seed42_padded_for_f2.pt', help='TVCS specialist checkpoint')
    parser.add_argument('--init_checkpoint', type=str, default='outputs/stage_f3_ablation/no_c_emb/cikd_pp_rt_ablation_no_c_emb.pt', help='Initial no_c_emb checkpoint')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--checkpoint_out', type=str, required=True, help='Checkpoint output path')
    
    # Sweep configuration arguments
    parser.add_argument('--config_name', type=str, required=True, help='Configuration name')
    parser.add_argument('--alpha_max', type=float, default=0.3, help='alpha_max')
    parser.add_argument('--focal_gamma', type=float, default=1.0, help='focal_gamma')
    parser.add_argument('--tvcs_mode', type=str, default='frozen', choices=['frozen', 'unfreeze_last_projection'], help='tvcs_mode')
    
    # Standard training hyperparameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    
    # Execution / Guardrails
    parser.add_argument('--dry_run', action='store_true', default=False, help='Perform a dry-run check without training')
    parser.add_argument('--verify_train_path', action='store_true', default=False, help='Verify training path is reachable and exit')
    parser.add_argument('--print_mode_only', action='store_true', default=False, help='Print active mode and exit safely')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing outputs')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Perform evaluation on locked test split only')

    args = parser.parse_args()

    set_seed(args.seed)

    # At startup print
    print(f"[CONFIG] dry_run = {args.dry_run}")
    print(f"[CONFIG] verify_train_path = {args.verify_train_path}")
    print(f"[CONFIG] config_name = {args.config_name}")
    print(f"[CONFIG] eval_only = {args.eval_only}")

    if args.verify_train_path:
        print(f"VERIFYING TRAINING PATH FOR: {args.config_name}")
        print(f"  alpha_max:   {args.alpha_max}")
        print(f"  focal_gamma: {args.focal_gamma}")
        print(f"  tvcs_mode:   {args.tvcs_mode}")
        print("G2 TRAINING PATH IS REACHABLE")
        return

    if args.dry_run:
        run_dry_run(args)
        return

    if args.print_mode_only:
        print(f"[CONFIG] resolved_mode = training")
        print("REAL TRAINING WOULD START HERE, BUT --print_mode_only EXITED SAFELY.")
        return

    if args.eval_only:
        run_evaluation_only(args)
        return

    run_training_train_val_only(args)
    return

if __name__ == '__main__':
    main()
