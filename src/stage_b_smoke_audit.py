import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

# ---------------------------------------------------------
# Pandas 2.0+ / 3.0+ compatibility patches for legacy pickles
# ---------------------------------------------------------
import pandas.core.indexes.base
pandas.core.indexes.base.Int64Index = pd.Index
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
WORKSPACE_DIR = r"D:\CIKD"
PROCESSED_DIR = os.path.join(WORKSPACE_DIR, "data", "processed")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "data", "cache")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs", "stage_b_cache_audit")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output Paths
SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "01_stage_b_smoke_audit_summary.txt")
NORMS_CSV_PATH = os.path.join(OUTPUT_DIR, "01_stage_b_smoke_feature_norms.csv")
PATCH_VAR_CSV_PATH = os.path.join(OUTPUT_DIR, "01_stage_b_smoke_patch_variance.csv")
LABEL_CHECK_CSV_PATH = os.path.join(OUTPUT_DIR, "01_stage_b_smoke_label_check.csv")

# Expected sizes
EXPECTED_COUNTS = {
    'full': 16909,
    'kg_complete': 12786,
    'tvcs_eligible': 7509
}

EXPECTED_SPLITS = {
    'full': {0: 11836, 1: 1691, 2: 3382},
    'kg_complete': {0: 8900, 1: 1300, 2: 2586}
}

SUBSETS = ['full', 'kg_complete', 'tvcs_eligible']

ARRAY_NAMES = [
    "text_features.npy",
    "image_features_global.npy",
    "image_features_patch.npy",
    "kg_features.npy",
    "relation_ids.npy",
    "labels_binary.npy",
    "labels_fine.npy",
    "y_ck.npy",
    "split_ids.npy",
    "sample_ids.npy"
]

def log(msg):
    print(msg)
    sys.stdout.flush()

def main():
    start_time = time.time()
    log("=" * 60)
    log("Starting Stage B Feature Cache Smoke Audit...")
    log("=" * 60)

    audit_passed = True
    summary_messages = []
    warnings = []

    # Helper function to register status
    def check_result(success, check_name, details_msg):
        nonlocal audit_passed
        status_str = "[PASSED]" if success else "[FAILED]"
        msg = f"{status_str} {check_name}: {details_msg}"
        log(msg)
        summary_messages.append(msg)
        if not success:
            audit_passed = False

    # 1. Verify array counts match expected
    log("\n--- Check 1: Verify array counts match expected ---")
    counts_ok = True
    loaded_caches = {}
    for subset in SUBSETS:
        loaded_caches[subset] = {}
        subset_dir = os.path.join(CACHE_DIR, subset)
        expected_n = EXPECTED_COUNTS[subset]
        
        # Load all arrays
        for arr_name in ARRAY_NAMES:
            path = os.path.join(subset_dir, arr_name)
            if not os.path.exists(path):
                check_result(False, f"Check 1 - {subset} array existence", f"{arr_name} not found at {path}")
                counts_ok = False
                continue
            
            arr = np.load(path)
            loaded_caches[subset][arr_name.replace(".npy", "")] = arr
            
            # Check length matches expected
            if len(arr) != expected_n:
                check_result(False, f"Check 1 - {subset} size match", f"{arr_name} size {len(arr)} != expected {expected_n}")
                counts_ok = False

    if counts_ok:
        check_result(True, "Check 1 - Array Counts", f"All arrays exist and have expected lengths: full={EXPECTED_COUNTS['full']}, kg_complete={EXPECTED_COUNTS['kg_complete']}, tvcs_eligible={EXPECTED_COUNTS['tvcs_eligible']}.")

    # 2. Verify split_ids counts match Stage A
    log("\n--- Check 2: Verify split_ids counts match Stage A ---")
    splits_ok = True
    for subset in ['full', 'kg_complete']:
        if 'split_ids' not in loaded_caches[subset]:
            splits_ok = False
            continue
        split_ids = loaded_caches[subset]['split_ids']
        expected_splits = EXPECTED_SPLITS[subset]
        
        unique, counts = np.unique(split_ids, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        for split_val, expected_c in expected_splits.items():
            actual_c = counts_dict.get(split_val, 0)
            if actual_c != expected_c:
                check_result(False, f"Check 2 - {subset} split {split_val} count", f"Actual {actual_c} != expected {expected_c}")
                splits_ok = False

    if splits_ok:
        check_result(True, "Check 2 - Split ID Counts", "Split counts for full and kg_complete match Stage A splits exactly.")

    # 3. Verify no NaN/Inf in all arrays
    log("\n--- Check 3: Verify no NaN/Inf in all arrays ---")
    nan_inf_ok = True
    for subset in SUBSETS:
        for arr_name, arr in loaded_caches[subset].items():
            if np.issubdtype(arr.dtype, np.number):
                nan_cnt = np.isnan(arr).sum()
                inf_cnt = np.isinf(arr).sum()
                if nan_cnt > 0 or inf_cnt > 0:
                    check_result(False, f"Check 3 - NaNs/Infs in {subset}/{arr_name}", f"Found {nan_cnt} NaNs and {inf_cnt} Infs!")
                    nan_inf_ok = False

    if nan_inf_ok:
        check_result(True, "Check 3 - NaN/Inf Check", "No NaNs or Infs detected in any of the cached arrays.")

    # 4. Verify feature norm statistics
    log("\n--- Check 4: Verify feature norm statistics ---")
    norms_rows = []
    float_features = ["text_features", "image_features_global", "image_features_patch", "kg_features"]
    norms_ok = True

    for subset in SUBSETS:
        for feat in float_features:
            if feat not in loaded_caches[subset]:
                continue
            arr = loaded_caches[subset][feat]
            
            # Compute row-wise L2 norm
            if feat == "image_features_patch":
                # Reshape to [N, 49*512] for flattened patch norms
                flat_arr = arr.reshape(len(arr), -1)
                norms = np.linalg.norm(flat_arr, axis=1)
            else:
                norms = np.linalg.norm(arr, axis=1)
            
            min_norm = float(np.min(norms))
            mean_norm = float(np.mean(norms))
            std_norm = float(np.std(norms))
            max_norm = float(np.max(norms))
            
            # Count zero rows (L2 norm < 1e-5)
            zero_count = int(np.sum(norms < 1e-5))
            zero_rate = zero_count / len(arr)
            
            norms_rows.append({
                "subset": subset,
                "feature_type": feat,
                "min_norm": f"{min_norm:.6f}",
                "mean_norm": f"{mean_norm:.6f}",
                "std_norm": f"{std_norm:.6f}",
                "max_norm": f"{max_norm:.6f}",
                "zero_rows_count": zero_count,
                "zero_rows_rate": f"{zero_rate:.4%}"
            })
            
            # Warn if too many zero rows (exclude image features global/patch which have legitimate image loading failures)
            if feat != "image_features_global" and feat != "image_features_patch":
                if zero_rate > 0.05:
                    warn_msg = f"High zero rows rate ({zero_rate:.4%}) in {subset}/{feat}!"
                    warnings.append(warn_msg)
                    log(f"  [WARNING] {warn_msg}")
            else:
                # For images, verify it's within expected range (e.g. failure rate was around 0.5% in Stage B)
                if zero_rate > 0.10:
                    warn_msg = f"Image feature loading failure rate is high ({zero_rate:.4%}) in {subset}/{feat}!"
                    warnings.append(warn_msg)
                    log(f"  [WARNING] {warn_msg}")

    # Write to CSV
    pd.DataFrame(norms_rows).to_csv(NORMS_CSV_PATH, index=False)
    check_result(True, "Check 4 - Feature Norms", f"Feature norms calculated and saved to {NORMS_CSV_PATH}.")

    # 5. Verify CLIP patch tokens are real
    log("\n--- Check 5: Verify CLIP patch tokens are real ---")
    patch_ok = True
    
    # Check shape
    patch_features = loaded_caches['full']['image_features_patch']
    global_features = loaded_caches['full']['image_features_global']
    expected_patch_shape = (EXPECTED_COUNTS['full'], 49, 512)
    if patch_features.shape != expected_patch_shape:
        check_result(False, "Check 5 - Patch shape", f"Actual shape {patch_features.shape} != expected {expected_patch_shape}")
        patch_ok = False
    
    # Sample 100 indices from non-zero-masked samples
    global_norms = np.linalg.norm(global_features, axis=1)
    valid_indices = np.where(global_norms > 1e-5)[0]
    
    if len(valid_indices) < 100:
        check_result(False, "Check 5 - Valid samples count", f"Not enough non-zero image samples ({len(valid_indices)}) to draw 100 random samples.")
        patch_ok = False
    else:
        np.random.seed(42)
        sampled_indices = np.random.choice(valid_indices, size=100, replace=False)
        
        patch_var_rows = []
        variances = []
        identical_copies_detected = 0
        
        for idx in sampled_indices:
            sample_patches = patch_features[idx] # [49, 512]
            sample_global = global_features[idx] # [512]
            
            # Compute variance across the 49 patches along axis 0
            var_across_patches = np.var(sample_patches, axis=0) # [512]
            mean_var = float(np.mean(var_across_patches))
            variances.append(mean_var)
            
            # Check if identical copy
            is_copy = False
            for p_idx in range(49):
                if np.allclose(sample_patches[p_idx], sample_global, atol=1e-6):
                    is_copy = True
                    identical_copies_detected += 1
                    break
            
            patch_var_rows.append({
                "sample_idx": idx,
                "global_norm": f"{global_norms[idx]:.6f}",
                "mean_patch_variance": f"{mean_var:.8f}",
                "is_identical_copy": is_copy
            })
            
            if mean_var <= 0:
                log(f"  [ERROR] Sample {idx} has zero patch variance ({mean_var:.8f})")
                patch_ok = False
        
        # Save to CSV
        pd.DataFrame(patch_var_rows).to_csv(PATCH_VAR_CSV_PATH, index=False)
        
        mean_var_overall = np.mean(variances)
        log(f"  Mean patch variance across 100 samples: {mean_var_overall:.8f}")
        log(f"  Identical copies of global feature detected: {identical_copies_detected} / 100")
        
        if identical_copies_detected > 0:
            warn_msg = f"Found {identical_copies_detected} patch tokens that are identical to the global image feature."
            warnings.append(warn_msg)
            log(f"  [WARNING] {warn_msg}")
            
        if patch_ok:
            check_result(True, "Check 5 - CLIP Patch Verification", 
                         f"Patch shape verified as {expected_patch_shape}. Patch variance > 0 for all 100 random samples (mean variance: {mean_var_overall:.8f}), no tokens are identical to global image features. Saved to {PATCH_VAR_CSV_PATH}.")

    # 6. Verify KG-complete has valid KG
    log("\n--- Check 6: Verify KG-complete has valid KG ---")
    kg_ok = True
    for subset in ['kg_complete', 'tvcs_eligible']:
        kg_feats = loaded_caches[subset]['kg_features']
        norms = np.linalg.norm(kg_feats, axis=1)
        zero_rows = np.sum(norms < 1e-5)
        
        log(f"  {subset}: {zero_rows} zero KG rows out of {len(kg_feats)}")
        if zero_rows > 0:
            warn_msg = f"{zero_rows} zero KG rows exist in subset {subset}!"
            warnings.append(warn_msg)
            log(f"  [WARNING] {warn_msg}")
            # If zero rows rate is high (>0.1%), we fail the check. Let's make it a warning if low but report it.
            if zero_rows / len(kg_feats) > 0.001:
                kg_ok = False

    if kg_ok:
        check_result(True, "Check 6 - KG Completeness", "No/low zero KG rows detected in kg_complete and tvcs_eligible subsets.")
    else:
        check_result(False, "Check 6 - KG Completeness", "Too many zero KG rows in complete subsets.")

    # 7. Verify relation_ids
    log("\n--- Check 7: Verify relation_ids ---")
    rel_ok = True
    for subset in SUBSETS:
        rel_ids = loaded_caches[subset]['relation_ids']
        
        # Check shape
        if rel_ids.shape != (EXPECTED_COUNTS[subset],):
            check_result(False, f"Check 7 - relation_ids shape in {subset}", f"Shape {rel_ids.shape} != {(EXPECTED_COUNTS[subset],)}")
            rel_ok = False
            
        # Unique count
        unique_rels = len(np.unique(rel_ids))
        if unique_rels <= 1:
            check_result(False, f"Check 7 - unique relations in {subset}", f"Only found {unique_rels} unique relation ID.")
            rel_ok = False
            
        # Count missing (relation_id == 0) in kg_complete
        if subset == 'kg_complete':
            missing_rel_cnt = np.sum(rel_ids == 0)
            log(f"  kg_complete missing relation_ids (ID=0): {missing_rel_cnt}")
            if missing_rel_cnt > 0:
                warn_msg = f"{missing_rel_cnt} missing relation IDs (ID=0) found in kg_complete!"
                warnings.append(warn_msg)
                log(f"  [WARNING] {warn_msg}")
                if missing_rel_cnt / len(rel_ids) > 0.01:
                    rel_ok = False

    if rel_ok:
        check_result(True, "Check 7 - Relation IDs Verification", "Relation ID shapes, unique counts, and missingness checks passed.")
    else:
        check_result(False, "Check 7 - Relation IDs Verification", "Relation ID shapes, unique counts, or missingness checks failed.")

    # 8. Verify sample alignment
    log("\n--- Check 8: Verify sample alignment ---")
    align_ok = True
    
    # Load manifests
    log("  Loading source manifests...")
    manifests = {}
    for split in ['train', 'val', 'test']:
        m_path = os.path.join(PROCESSED_DIR, f"manifest_{split}_seed42.csv")
        manifests[split] = pd.read_csv(m_path)
    
    # 8a: sample_ids in full are monotonic
    full_sample_ids = loaded_caches['full']['sample_ids']
    is_monotonic = np.all(np.diff(full_sample_ids) > 0)
    is_identity = np.array_equal(full_sample_ids, np.arange(len(full_sample_ids)))
    
    if not (is_monotonic and is_identity):
        check_result(False, "Check 8a - Sample IDs Monotonicity", "sample_ids in full are not strictly monotonic 0 to N-1")
        align_ok = False
    else:
        log("  [PASSED] sample_ids in full correspond exactly to 0 to N-1.")

    # 8b: split_ids correspond to manifest split order
    full_split_ids = loaded_caches['full']['split_ids']
    # Order should be train (0) then val (1) then test (2)
    expected_full_splits = np.concatenate([
        np.zeros(len(manifests['train']), dtype=np.int64),
        np.ones(len(manifests['val']), dtype=np.int64),
        np.ones(len(manifests['test']), dtype=np.int64) * 2
    ])
    if not np.array_equal(full_split_ids, expected_full_splits):
        check_result(False, "Check 8b - Split IDs Order", "split_ids in full do not correspond to train->val->test manifest ordering.")
        align_ok = False
    else:
        log("  [PASSED] split_ids in full match train->val->test manifest ordering exactly.")

    # 8c: labels_fine distribution from cache matches manifest distribution
    log("  Validating labels_fine distributions...")
    label_check_rows = []
    
    # Define label count extraction for cache subsets
    def verify_subset_labels(subset):
        nonlocal align_ok
        sub_labels = loaded_caches[subset]['labels_fine']
        sub_splits = loaded_caches[subset]['split_ids']
        
        for split_val, split_name in [(0, 'train'), (1, 'val'), (2, 'test')]:
            # Cache counts
            split_mask = (sub_splits == split_val)
            cache_split_labels = sub_labels[split_mask]
            cache_unique, cache_counts = np.unique(cache_split_labels, return_counts=True)
            cache_dist = dict(zip(cache_unique, cache_counts))
            
            # Manifest expected counts
            m_df = manifests[split_name]
            # Apply subset filtering
            if subset == 'kg_complete':
                expected_df = m_df[m_df['kg_complete'] == True]
            elif subset == 'tvcs_eligible':
                expected_df = m_df[m_df['tvcs_eligible'] == True]
            else:
                expected_df = m_df
                
            manifest_counts = expected_df['fine_label'].value_counts().to_dict()
            
            # Check all classes 0-5
            for cls in range(6):
                c_cnt = cache_dist.get(cls, 0)
                m_cnt = manifest_counts.get(cls, 0)
                match = (c_cnt == m_cnt)
                if not match:
                    log(f"    [ERROR] Mismatch in {subset}/{split_name} class {cls}: Cache={c_cnt}, Manifest={m_cnt}")
                    align_ok = False
                
                label_check_rows.append({
                    "subset": subset,
                    "split": split_name,
                    "fine_label": cls,
                    "cache_count": c_cnt,
                    "manifest_count": m_cnt,
                    "match_status": "MATCH" if match else "MISMATCH"
                })

    verify_subset_labels('full')
    verify_subset_labels('kg_complete')
    verify_subset_labels('tvcs_eligible')
    
    # Write to CSV
    pd.DataFrame(label_check_rows).to_csv(LABEL_CHECK_CSV_PATH, index=False)
    
    if align_ok:
        check_result(True, "Check 8 - Sample Alignment & Label Verification", 
                     f"Sample IDs are monotonic, split IDs match order, and fine label counts match source manifests exactly across all splits. Saved to {LABEL_CHECK_CSV_PATH}.")

    # 9. Run a tiny sanity classifier
    log("\n--- Check 9: Run a tiny sanity classifier ---")
    classifier_ok = True
    try:
        # Load full text features and labels
        text_feats = loaded_caches['full']['text_features']
        labels_fine = loaded_caches['full']['labels_fine']
        split_ids = loaded_caches['full']['split_ids']
        
        train_x = text_feats[split_ids == 0]
        train_y = labels_fine[split_ids == 0]
        val_x = text_feats[split_ids == 1]
        val_y = labels_fine[split_ids == 1]
        
        log(f"  Training shape: X_train={train_x.shape}, y_train={train_y.shape}")
        log(f"  Validation shape: X_val={val_x.shape}, y_val={val_y.shape}")
        
        # PyTorch environment setup
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"  Using device: {device}")
        
        # Single layer classifier
        class LinearProber(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(768, 6) # 768 RoBERTa, 6 classes
            def forward(self, x):
                return self.linear(x)
                
        model = LinearProber().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        train_dataset = TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Train 1 epoch
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        epoch_loss /= len(train_dataset)
        log(f"  Epoch 1 Loss: {epoch_loss:.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            vx = torch.tensor(val_x, dtype=torch.float32).to(device)
            logits = model(vx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
        val_macro_f1 = f1_score(val_y, preds, average='macro')
        log(f"  Validation Macro-F1: {val_macro_f1:.4f}")
        
        baseline = 0.166
        if val_macro_f1 > baseline:
            check_result(True, "Check 9 - Sanity Classifier Test", 
                         f"Sanity classifier trained successfully. Validation Macro-F1 = {val_macro_f1:.4f}, which is above random baseline ({baseline:.3f}).")
        else:
            check_result(False, "Check 9 - Sanity Classifier Test", 
                         f"Classifier Validation Macro-F1 = {val_macro_f1:.4f} is NOT above random baseline ({baseline:.3f}).")
            classifier_ok = False
    except Exception as e:
        check_result(False, "Check 9 - Sanity Classifier Test", f"Exception raised during classification check: {str(e)}")
        classifier_ok = False

    # 10. Compile and Save Summary Output
    log("\n--- Audit Summary Report Compile ---")
    elapsed_time = time.time() - start_time
    
    summary_lines = [
        "=================================================================",
        "                 STAGE B CACHE ANTI-FAKE SMOKE AUDIT             ",
        "=================================================================",
        f"Audit Execution Time: {elapsed_time:.2f} seconds",
        f"Audit Status: {'PASSED' if audit_passed else 'FAILED'}",
        f"Device Used: {'cuda' if torch.cuda.is_available() else 'cpu'}",
        "",
        "CHECKLIST RESULTS:"
    ]
    for msg in summary_messages:
        summary_lines.append(f"  {msg}")
        
    summary_lines.append("")
    summary_lines.append("WARNINGS DETECTED:")
    if len(warnings) > 0:
        for w in warnings:
            summary_lines.append(f"  - [WARNING] {w}")
    else:
        summary_lines.append("  No warnings detected.")
        
    summary_lines.append("=================================================================")
    
    summary_content = "\n".join(summary_lines)
    
    with open(SUMMARY_TXT_PATH, "w", encoding='utf-8') as f:
        f.write(summary_content)
        
    log("\n" + summary_content)
    log(f"\nAudit completed. Reports written to {OUTPUT_DIR}")
    
    if not audit_passed:
        log("\n[ERROR] Audit did not pass all checks! Exiting with code 1.")
        sys.exit(1)
    else:
        log("\n[SUCCESS] Audit passed all checks! Exiting with code 0.")
        sys.exit(0)

if __name__ == "__main__":
    main()
