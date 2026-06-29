import os
import sys
import time
import json
import ast
import gc
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel, CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm

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
AUDIT_DIR = os.path.join(WORKSPACE_DIR, "outputs", "stage_b_cache_audit")

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helper functions for KG parsing & relation extraction
# ---------------------------------------------------------
def parse_kg_embedding(val):
    if pd.isnull(val):
        return np.zeros(100, dtype=np.float32)
    val_str = str(val).strip()
    if val_str == "" or val_str == "[]" or val_str == "None":
        return np.zeros(100, dtype=np.float32)
    cleaned = val_str.replace('[', '').replace(']', '').replace('\n', ' ').strip()
    try:
        parts = [float(x) for x in cleaned.split() if x.strip()]
        if len(parts) == 0:
            return np.zeros(100, dtype=np.float32)
        if len(parts) != 100:
            arr = np.zeros(100, dtype=np.float32)
            arr[:min(len(parts), 100)] = parts[:min(len(parts), 100)]
            return arr
        return np.array(parts, dtype=np.float32)
    except ValueError:
        return np.zeros(100, dtype=np.float32)

def extract_relations(val):
    if pd.isnull(val):
        return []
    val_str = str(val).strip()
    if val_str == "" or val_str == "[]" or val_str == "None":
        return []
    try:
        parsed = ast.literal_eval(val_str)
        rels = []
        if isinstance(parsed, list):
            for entity_list in parsed:
                if isinstance(entity_list, list):
                    for triple in entity_list:
                        if isinstance(triple, list) and len(triple) >= 2:
                            rels.append(triple[1])
        return rels
    except Exception:
        return []

def load_image_safe(path, failures_list, index_in_full):
    if not isinstance(path, str) or not path.strip():
        failures_list.append({"index": index_in_full, "path": "", "reason": "Empty path"})
        return None
    if not os.path.exists(path):
        failures_list.append({"index": index_in_full, "path": path, "reason": "File does not exist"})
        return None
    try:
        with Image.open(path) as img:
            loaded_img = img.convert('RGB')
            loaded_img.load()  # Force load pixel data
            return loaded_img
    except Exception as e:
        failures_list.append({"index": index_in_full, "path": path, "reason": f"Open error: {str(e)}"})
        return None

# ---------------------------------------------------------
# Main execution pipeline
# ---------------------------------------------------------
def main():
    start_time = time.time()
    print("=" * 60)
    print("Starting Stage B Feature Caching Pipeline (Memory Optimized)...")
    print("=" * 60)

    # 1. Load manifests
    print("1. Loading manifests...")
    splits = ['train', 'val', 'test']
    dfs = {}
    for split in splits:
        path = os.path.join(PROCESSED_DIR, f"manifest_{split}_seed42.csv")
        if not os.path.exists(path):
            print(f"ERROR: Manifest not found at {path}")
            sys.exit(1)
        dfs[split] = pd.read_csv(path)
        print(f"  Loaded {split} split: {len(dfs[split])} rows")

    # Add split_id (0=train, 1=val, 2=test)
    dfs['train']['split_id'] = 0
    dfs['val']['split_id'] = 1
    dfs['test']['split_id'] = 2

    # Concatenate to unified full dataset
    df_full = pd.concat([dfs['train'], dfs['val'], dfs['test']], ignore_index=True)
    df_full['sample_id'] = np.arange(len(df_full))
    N_full = len(df_full)
    print(f"Unified full dataset size: {N_full} rows")

    # 2. Extract relations and build vocab
    print("\n2. Building relation vocabulary and relation_id...")
    all_sample_rels = []
    for idx, row in df_full.iterrows():
        all_sample_rels.append(extract_relations(row['relation']))

    # Extract unique relations (ignoring empty lists)
    unique_relations = sorted(list(set(rel for rels in all_sample_rels for rel in rels)))
    # We reserve 0 for <pad> (no relations), and 1 for <unk> (unseen relations, if any)
    vocab = {'<pad>': 0, '<unk>': 1}
    for rel in unique_relations:
        if rel not in vocab:
            vocab[rel] = len(vocab)

    # Map each sample's first relation to its ID, default to 0 (<pad>) if empty
    relation_ids = []
    for rels in all_sample_rels:
        if len(rels) > 0:
            first_rel = rels[0]
            relation_ids.append(vocab.get(first_rel, 1))
        else:
            relation_ids.append(0)
    relation_ids = np.array(relation_ids, dtype=np.int64)

    # Save vocabulary
    vocab_path = os.path.join(CACHE_DIR, "relation_vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=4)
    print(f"Saved relation vocabulary to {vocab_path} (size: {len(vocab)})")

    # 3. RoBERTa Text Feature Extraction (CUDA, batched)
    print("\n3. Extracting RoBERTa text features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for text features: {device}")
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model_text = RobertaModel.from_pretrained("roberta-base").to(device)
    model_text.eval()

    batch_texts = list(df_full['text'].fillna("").values)
    text_features = []
    text_batch_size = 64  # Reduced batch size to lower VRAM overhead
    
    with torch.no_grad():
        for i in tqdm(range(0, len(batch_texts), text_batch_size), desc="RoBERTa Extractor"):
            batch = batch_texts[i : i + text_batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model_text(**inputs)
            # Extracted CLS tokens
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            text_features.append(cls_embeds)
            
    text_features = np.concatenate(text_features, axis=0)
    print(f"RoBERTa text features extracted: shape {text_features.shape}")

    # Explicit memory cleanup of RoBERTa model to free VRAM for CLIP
    print("Cleaning up RoBERTa memory context...")
    del model_text
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)  # Give WDDM driver time to release memory

    # 4. CLIP Image Feature Extraction (CUDA, batched)
    print("\n4. Extracting CLIP image features...")
    print(f"Device for image features: {device}")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model_clip.eval()

    batch_paths = list(df_full['abs_image_path'].values)
    image_features_global = []
    image_features_patch = []
    image_failures = []
    
    image_batch_size = 32  # Reduced batch size to stay safely within 4GB VRAM
    with torch.no_grad():
        for i in tqdm(range(0, len(batch_paths), image_batch_size), desc="CLIP Extractor"):
            batch_p = batch_paths[i : i + image_batch_size]
            batch_images = []
            failed_indices_in_batch = []
            
            for offset, path in enumerate(batch_p):
                idx_in_full = i + offset
                img = load_image_safe(path, image_failures, idx_in_full)
                if img is None:
                    # Create placeholder black image
                    img = Image.new('RGB', (224, 224), color=0)
                    failed_indices_in_batch.append(offset)
                batch_images.append(img)
            
            # Preprocess the batch
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            vision_outputs = model_clip.vision_model(**inputs)
            
            # Global representation projected to 512 dimensions
            global_embeds = model_clip.visual_projection(vision_outputs[1]) # [B, 512]
            
            # Patch representations (CLS removed, shape [B, 49, 768]) projected to 512 dimensions
            patch_embeds = model_clip.visual_projection(vision_outputs[0][:, 1:, :]) # [B, 49, 512]
            
            global_embeds = global_embeds.cpu().numpy()
            patch_embeds = patch_embeds.cpu().numpy()
            
            # Mask failed images to all zeros
            for offset in failed_indices_in_batch:
                global_embeds[offset] = 0.0
                patch_embeds[offset] = 0.0
            
            image_features_global.append(global_embeds)
            image_features_patch.append(patch_embeds)

    image_features_global = np.concatenate(image_features_global, axis=0)
    image_features_patch = np.concatenate(image_features_patch, axis=0)
    print(f"CLIP global features shape: {image_features_global.shape}")
    print(f"CLIP patch features shape: {image_features_patch.shape}")
    print(f"Image failures: {len(image_failures)} out of {N_full}")

    # Explicit memory cleanup of CLIP model
    print("Cleaning up CLIP memory context...")
    del model_clip
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Extract KG features
    print("\n5. Parsing KG embeddings...")
    kg_features = []
    for val in df_full['knowledge_embedding']:
        kg_features.append(parse_kg_embedding(val))
    kg_features = np.stack(kg_features, axis=0)
    print(f"KG features shape: {kg_features.shape}")

    # 6. Extract Labels and Metadata
    print("\n6. Extracting labels and metadata...")
    labels_binary = df_full['label'].values.astype(np.int64)
    labels_fine = df_full['fine_label'].values.astype(np.int64)
    y_ck = df_full['y_ck'].values.astype(np.int64)
    split_ids = df_full['split_id'].values.astype(np.int64)
    sample_ids = df_full['sample_id'].values.astype(np.int64)

    # 7. Masking & Slicing into Cache Sets
    print("\n7. Slicing and saving cache sets...")
    mask_full = np.ones(N_full, dtype=bool)
    mask_kg = (df_full['kg_complete'] == True).values
    mask_tvcs = (df_full['tvcs_eligible'] == True).values

    cache_sets = {
        'full': mask_full,
        'kg_complete': mask_kg,
        'tvcs_eligible': mask_tvcs
    }

    # Verify subset shapes
    for name, mask in cache_sets.items():
        set_dir = os.path.join(CACHE_DIR, name)
        os.makedirs(set_dir, exist_ok=True)

        # Slice arrays
        sub_text = text_features[mask]
        sub_img_global = image_features_global[mask]
        sub_img_patch = image_features_patch[mask]
        sub_kg = kg_features[mask]
        sub_rel_ids = relation_ids[mask]
        sub_lbl_bin = labels_binary[mask]
        sub_lbl_fine = labels_fine[mask]
        sub_y_ck = y_ck[mask]
        sub_split = split_ids[mask]
        sub_sample = sample_ids[mask]

        # Save arrays
        np.save(os.path.join(set_dir, "text_features.npy"), sub_text)
        np.save(os.path.join(set_dir, "image_features_global.npy"), sub_img_global)
        np.save(os.path.join(set_dir, "image_features_patch.npy"), sub_img_patch)
        np.save(os.path.join(set_dir, "kg_features.npy"), sub_kg)
        np.save(os.path.join(set_dir, "relation_ids.npy"), sub_rel_ids)
        np.save(os.path.join(set_dir, "labels_binary.npy"), sub_lbl_bin)
        np.save(os.path.join(set_dir, "labels_fine.npy"), sub_lbl_fine)
        np.save(os.path.join(set_dir, "y_ck.npy"), sub_y_ck)
        np.save(os.path.join(set_dir, "split_ids.npy"), sub_split)
        np.save(os.path.join(set_dir, "sample_ids.npy"), sub_sample)

        print(f"  Saved cache set '{name}' ({mask.sum()} samples) to {set_dir}")

    # 8. Run Critical Verification Checks
    print("\n8. Running critical verification checks...")
    shapes_rows = []
    norms_rows = []
    nan_inf_rows = []

    for name, mask in cache_sets.items():
        set_dir = os.path.join(CACHE_DIR, name)
        
        # Load and verify
        t_feat = np.load(os.path.join(set_dir, "text_features.npy"))
        ig_feat = np.load(os.path.join(set_dir, "image_features_global.npy"))
        ip_feat = np.load(os.path.join(set_dir, "image_features_patch.npy"))
        k_feat = np.load(os.path.join(set_dir, "kg_features.npy"))
        r_ids = np.load(os.path.join(set_dir, "relation_ids.npy"))
        lbl_b = np.load(os.path.join(set_dir, "labels_binary.npy"))
        lbl_f = np.load(os.path.join(set_dir, "labels_fine.npy"))
        y_c = np.load(os.path.join(set_dir, "y_ck.npy"))
        spl = np.load(os.path.join(set_dir, "split_ids.npy"))
        sam = np.load(os.path.join(set_dir, "sample_ids.npy"))

        N = mask.sum()

        # Check shapes
        assert t_feat.shape == (N, 768), f"Error: text shape {t_feat.shape} != {(N, 768)}"
        assert ig_feat.shape == (N, 512), f"Error: image global shape {ig_feat.shape} != {(N, 512)}"
        assert ip_feat.shape == (N, 49, 512), f"Error: image patch shape {ip_feat.shape} != {(N, 49, 512)}"
        assert k_feat.shape == (N, 100), f"Error: KG shape {k_feat.shape} != {(N, 100)}"
        assert r_ids.shape == (N,), f"Error: relation_id shape {r_ids.shape} != {(N,)}"
        assert lbl_b.shape == (N,), f"Error: labels_binary shape {lbl_b.shape} != {(N,)}"
        assert lbl_f.shape == (N,), f"Error: labels_fine shape {lbl_f.shape} != {(N,)}"
        assert y_c.shape == (N,), f"Error: y_ck shape {y_c.shape} != {(N,)}"
        assert spl.shape == (N,), f"Error: split_ids shape {spl.shape} != {(N,)}"
        assert sam.shape == (N,), f"Error: sample_ids shape {sam.shape} != {(N,)}"

        # Save to shape report rows
        arrays_to_report = [
            ("text_features", t_feat),
            ("image_features_global", ig_feat),
            ("image_features_patch", ip_feat),
            ("kg_features", k_feat),
            ("relation_ids", r_ids),
            ("labels_binary", lbl_b),
            ("labels_fine", lbl_f),
            ("y_ck", y_c),
            ("split_ids", spl),
            ("sample_ids", sam)
        ]
        for arr_name, arr in arrays_to_report:
            shapes_rows.append({
                "cache_set": name,
                "feature_type": arr_name,
                "shape": str(list(arr.shape))
            })

        # Save to norm report rows (for floats)
        float_arrays = [
            ("text_features", t_feat),
            ("image_features_global", ig_feat),
            ("image_features_patch", ip_feat),
            ("kg_features", k_feat)
        ]
        for arr_name, arr in float_arrays:
            # L2 norm along last dim
            if arr_name == "image_features_patch":
                flat_arr = arr.reshape(len(arr), -1)
                l2_norms = np.linalg.norm(flat_arr, axis=-1)
            else:
                l2_norms = np.linalg.norm(arr, axis=-1)
                
            norms_rows.append({
                "cache_set": name,
                "feature_type": arr_name,
                "mean": f"{np.mean(arr):.6f}",
                "std": f"{np.std(arr):.6f}",
                "min": f"{np.min(arr):.6f}",
                "max": f"{np.max(arr):.6f}",
                "l2_norm_mean": f"{np.mean(l2_norms):.6f}"
            })

        # Save to nan/inf report rows
        for arr_name, arr in arrays_to_report:
            nan_cnt = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
            inf_cnt = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
            nan_inf_rows.append({
                "cache_set": name,
                "feature_type": arr_name,
                "nan_count": nan_cnt,
                "inf_count": inf_cnt
            })
            
            # Assert no NaNs or Infs
            assert nan_cnt == 0, f"ERROR: {arr_name} in set {name} contains {nan_cnt} NaNs!"
            assert inf_cnt == 0, f"ERROR: {arr_name} in set {name} contains {inf_cnt} Infs!"

        print(f"  All verification checks passed for set '{name}'.")

    # 9. Save Audit Files
    print("\n9. Saving audit files...")
    pd.DataFrame(shapes_rows).to_csv(os.path.join(AUDIT_DIR, "01_feature_shapes.csv"), index=False)
    pd.DataFrame(norms_rows).to_csv(os.path.join(AUDIT_DIR, "01_feature_norms.csv"), index=False)
    pd.DataFrame(nan_inf_rows).to_csv(os.path.join(AUDIT_DIR, "01_nan_inf_report.csv"), index=False)

    # Save summary report text
    elapsed_time = time.time() - start_time
    summary_lines = [
        "=================================================================",
        "                    STAGE B FEATURE CACHE SUMMARY                ",
        "=================================================================",
        f"CUDA Available: {torch.cuda.is_available()}",
        f"CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        f"Total Execution Time: {elapsed_time:.2f} seconds",
        f"Unified full dataset size: {N_full} samples",
        "",
        "CACHE SET SIZES:",
        f"  - Full:         {mask_full.sum()} samples",
        f"  - KG Complete:  {mask_kg.sum()} samples",
        f"  - TVCS Eligible: {mask_tvcs.sum()} samples",
        "",
        "RELATION VOCABULARY STATISTICS:",
        f"  - Vocab Size: {len(vocab)}",
        f"  - Vocab Path: {vocab_path}",
        "",
        "IMAGE FEATURE EXTRACTION FAILURES REPORT:",
        f"  - Total Failed: {len(image_failures)}",
        f"  - Failure Rate: {len(image_failures)/N_full:.4%}",
        "",
        "FAILED IMAGE DETAILS:"
    ]
    if len(image_failures) > 0:
        for fail in image_failures[:50]:
            summary_lines.append(f"  - Index {fail['index']} | Path: {fail['path']} | Reason: {fail['reason']}")
        if len(image_failures) > 50:
            summary_lines.append(f"  - ... and {len(image_failures) - 50} more failures (see code or log)")
    else:
        summary_lines.append("  No failures detected. All images loaded successfully.")
    
    summary_lines.append("=================================================================")
    summary_content = "\n".join(summary_lines)
    
    with open(os.path.join(AUDIT_DIR, "01_cache_summary.txt"), "w", encoding='utf-8') as f:
        f.write(summary_content)

    print("\nAudit reports saved successfully to:", AUDIT_DIR)
    print(summary_content)
    print("\nStage B Feature Caching Pipeline completed successfully!")

if __name__ == "__main__":
    main()
