import os
import sys
import numpy as np

def test_shape_and_numeric(array_path, expected_shape):
    if not os.path.exists(array_path):
        print(f"  [MISSING] {array_path}")
        return False
    try:
        # Load with mmap to avoid loading large files into memory
        arr = np.load(array_path, mmap_mode='r')
        actual_shape = arr.shape
        if actual_shape != expected_shape:
            print(f"  [FAIL] {os.path.basename(array_path)}: Shape mismatch. Expected {expected_shape}, got {actual_shape}")
            return False
        
        # Check NaN/Inf on first 1000 rows (or total rows if less)
        check_len = min(1000, len(arr))
        sample = arr[:check_len]
        nan_count = np.isnan(sample).sum()
        inf_count = np.isinf(sample).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  [FAIL] {os.path.basename(array_path)}: Found {nan_count} NaNs and {inf_count} Infs in sample of {check_len} rows.")
            return False
            
        print(f"  [PASS] {os.path.basename(array_path)}: Shape {actual_shape} is correct. Sample checked for NaNs/Infs.")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to load {array_path}: {e}")
        return False

def main():
    print("=" * 60)
    print("VERIFY STAGECD PACKAGE - PYTHON SHAPE AND NUMERIC VERIFIER")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_root = os.path.dirname(script_dir)
    print(f"Detected Target Root: {target_root}")
    
    cache_root = os.path.join(target_root, "data", "cache")
    if not os.path.exists(cache_root):
        print(f"[ERROR] Cache directory not found at: {cache_root}")
        print("VERIFY_STAGECD_PACKAGE_FAIL")
        sys.exit(1)
        
    expected_shapes = {
        'full': {
            'text_features.npy': (16909, 768),
            'image_features_global.npy': (16909, 512),
            'image_features_patch.npy': (16909, 49, 512),
            'kg_features.npy': (16909, 100),
        },
        'kg_complete': {
            'text_features.npy': (12786, 768),
            'image_features_patch.npy': (12786, 49, 512),
            'kg_features.npy': (12786, 100),
        },
        'tvcs_eligible': {
            'text_features.npy': (7509, 768),
            'kg_features.npy': (7509, 100),
        }
    }
    
    all_passed = True
    
    for subset, files in expected_shapes.items():
        print(f"\nVerifying subset: {subset}")
        subset_dir = os.path.join(cache_root, subset)
        for fname, shape in files.items():
            fpath = os.path.join(subset_dir, fname)
            success = test_shape_and_numeric(fpath, shape)
            if not success:
                all_passed = False
                
    # Verify split_ids contains 0, 1, 2
    print("\nVerifying split_ids distribution...")
    for subset in ['full', 'kg_complete', 'tvcs_eligible']:
        split_path = os.path.join(cache_root, subset, 'split_ids.npy')
        if os.path.exists(split_path):
            try:
                splits = np.load(split_path, mmap_mode='r')
                unique_splits = np.unique(splits)
                expected_splits = {0, 1, 2}
                actual_splits = set(unique_splits.tolist())
                if expected_splits.issubset(actual_splits) or actual_splits == expected_splits:
                    print(f"  [PASS] {subset}/split_ids.npy contains expected splits: {actual_splits}")
                else:
                    print(f"  [FAIL] {subset}/split_ids.npy contains splits: {actual_splits}, expected 0, 1, 2")
                    all_passed = False
            except Exception as e:
                print(f"  [ERROR] Failed to check splits for {subset}: {e}")
                all_passed = False
        else:
            print(f"  [MISSING] {split_path}")
            all_passed = False
            
    print("\n" + "=" * 60)
    if all_passed:
        print("VERIFY_STAGECD_PACKAGE_PASS")
        print("=" * 60)
        sys.exit(0)
    else:
        print("VERIFY_STAGECD_PACKAGE_FAIL")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
