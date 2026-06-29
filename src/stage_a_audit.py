import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
RAW_PICKLE_PATH = os.path.join(WORKSPACE_DIR, "data", "raw", "FineFake", "FineFake.pkl")
RAW_IMAGE_DIR = os.path.join(WORKSPACE_DIR, "data", "raw", "FineFake", "Image")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs", "stage_a_audit")
PROCESSED_DIR = os.path.join(WORKSPACE_DIR, "data", "processed")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Starting Stage A Data Audit...")
print(f"Pickle path: {RAW_PICKLE_PATH}")
print(f"Image directory: {RAW_IMAGE_DIR}")

# 1. Load FineFake.pkl
if not os.path.exists(RAW_PICKLE_PATH):
    print(f"ERROR: Legacy pickle file not found at {RAW_PICKLE_PATH}")
    sys.exit(1)

try:
    with open(RAW_PICKLE_PATH, 'rb') as f:
        df = pickle.load(f)
except Exception as e:
    print(f"ERROR: Failed to load legacy pickle file. Reason: {e}")
    sys.exit(1)

# 2. Print Shape
print(f"Dataframe shape: {df.shape}")

# 3. Detect and Normalize Columns
TARGET_COLUMNS = [
    'text', 'image_path', 'entity_id', 'topic', 'label', 
    'fine-grained label', 'knowledge_embedding', 'description', 
    'relation', 'platform', 'author', 'date', 'comment'
]

# Case-insensitive column resolution
current_columns = df.columns.tolist()
rename_map = {}
for col in current_columns:
    col_normalized = col.strip().lower().replace('_', '').replace('-', '').replace(' ', '')
    for target in TARGET_COLUMNS:
        target_normalized = target.lower().replace('_', '').replace('-', '').replace(' ', '')
        if col_normalized == target_normalized:
            rename_map[col] = target
            break

df = df.rename(columns=rename_map)

# Check for missing expected columns
missing_columns = [col for col in TARGET_COLUMNS if col not in df.columns]
if missing_columns:
    print(f"ERROR: Missing expected columns after normalization: {missing_columns}")
    sys.exit(1)

# Reorder columns to match TARGET_COLUMNS
df = df[TARGET_COLUMNS].copy()

# Save Schema
schema_rows = []
for col in df.columns:
    dtype = str(df[col].dtype)
    # Get a clean sample value representation
    if len(df) > 0:
        sample = df[col].iloc[0]
        sample_str = str(sample)
        if len(sample_str) > 100:
            sample_str = sample_str[:97] + "..."
    else:
        sample_str = "None"
    schema_rows.append({
        'column_name': col,
        'data_type': dtype,
        'sample_value': sample_str
    })
schema_df = pd.DataFrame(schema_rows)
schema_df.to_csv(os.path.join(OUTPUT_DIR, "00_dataset_schema.csv"), index=False)

# 4 & 5. Build absolute image paths and check existence
def resolve_abs_path(p):
    if pd.isnull(p):
        return ""
    clean_p = str(p).replace('\\', '/')
    if clean_p.startswith('Image/'):
        clean_p = clean_p[6:]
    return os.path.join(RAW_IMAGE_DIR, clean_p.replace('/', os.sep))

df['abs_image_path'] = df['image_path'].apply(resolve_abs_path)
df['image_exists'] = df['abs_image_path'].apply(lambda x: os.path.exists(x) if x else False)

# 6. Report statistics
total_samples = len(df)

# Unique image paths referenced in df
total_unique_image_paths = df['image_path'].nunique()

# Count physical files in raw image folder
def count_physical_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

total_physical_images = count_physical_files(RAW_IMAGE_DIR)

# Missing rates functions
def is_missing_str(x):
    return pd.isnull(x) or (isinstance(x, str) and len(x.strip()) == 0)

def is_missing_list(x):
    if isinstance(x, (list, np.ndarray, set)):
        return len(x) == 0
    return pd.isnull(x)

def is_missing_embed(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return True
        return all(v == 0.0 for v in x)
    return pd.isnull(x)

missing_text_count = df['text'].apply(is_missing_str).sum()
missing_text_rate = missing_text_count / total_samples

missing_image_path_count = df['image_path'].apply(is_missing_str).sum()
missing_image_path_rate = missing_image_path_count / total_samples

image_exists_count = df['image_exists'].sum()
image_exists_rate = image_exists_count / total_samples

missing_entity_id_count = df['entity_id'].apply(is_missing_list).sum()
missing_entity_id_rate = missing_entity_id_count / total_samples

missing_knowledge_embedding_count = df['knowledge_embedding'].apply(is_missing_embed).sum()
missing_knowledge_embedding_rate = missing_knowledge_embedding_count / total_samples

missing_relation_count = df['relation'].apply(is_missing_list).sum()
missing_relation_rate = missing_relation_count / total_samples

# Create missing fields report dataframe
missing_report = pd.DataFrame([
    {'column_name': 'text', 'missing_count': missing_text_count, 'missing_rate': f"{missing_text_rate:.4%}"},
    {'column_name': 'image_path', 'missing_count': missing_image_path_count, 'missing_rate': f"{missing_image_path_rate:.4%}"},
    {'column_name': 'entity_id', 'missing_count': missing_entity_id_count, 'missing_rate': f"{missing_entity_id_rate:.4%}"},
    {'column_name': 'knowledge_embedding', 'missing_count': missing_knowledge_embedding_count, 'missing_rate': f"{missing_knowledge_embedding_rate:.4%}"},
    {'column_name': 'relation', 'missing_count': missing_relation_count, 'missing_rate': f"{missing_relation_rate:.4%}"}
])
missing_report.to_csv(os.path.join(OUTPUT_DIR, "00_missing_fields_report.csv"), index=False)

# 7. Labels fixed mapping
BINARY_LABEL_MAP = {0: 'fake', 1: 'real'}
FINE_LABEL_MAP = {
    0: 'real',
    1: 'text-image inconsistency',
    2: 'content-knowledge inconsistency',
    3: 'text-based fake',
    4: 'image-based fake',
    5: 'others'
}

# Label distributions
label_counts = df['label'].value_counts()
label_dist = pd.DataFrame({
    'label': label_counts.index,
    'label_name': label_counts.index.map(BINARY_LABEL_MAP),
    'count': label_counts.values,
    'percentage': (label_counts.values / total_samples).astype(float)
}).sort_values('label')
label_dist['percentage'] = label_dist['percentage'].apply(lambda x: f"{x:.4%}")
label_dist.to_csv(os.path.join(OUTPUT_DIR, "00_label_distribution.csv"), index=False)

fine_label_counts = df['fine-grained label'].value_counts()
fine_label_dist = pd.DataFrame({
    'fine_label': fine_label_counts.index,
    'fine_label_name': fine_label_counts.index.map(FINE_LABEL_MAP),
    'count': fine_label_counts.values,
    'percentage': (fine_label_counts.values / total_samples).astype(float)
}).sort_values('fine_label')
fine_label_dist['percentage'] = fine_label_dist['percentage'].apply(lambda x: f"{x:.4%}")
fine_label_dist.to_csv(os.path.join(OUTPUT_DIR, "00_fine_label_distribution.csv"), index=False)

# Topic distribution
topic_counts = df['topic'].value_counts()
topic_dist = pd.DataFrame({
    'topic': topic_counts.index,
    'count': topic_counts.values,
    'percentage': (topic_counts.values / total_samples).astype(float)
})
topic_dist['percentage'] = topic_dist['percentage'].apply(lambda x: f"{x:.4%}")
topic_dist.to_csv(os.path.join(OUTPUT_DIR, "00_topic_distribution.csv"), index=False)

# Platform distribution
platform_counts = df['platform'].value_counts()
platform_dist = pd.DataFrame({
    'platform': platform_counts.index,
    'count': platform_counts.values,
    'percentage': (platform_counts.values / total_samples).astype(float)
})
platform_dist['percentage'] = platform_dist['percentage'].apply(lambda x: f"{x:.4%}")
platform_dist.to_csv(os.path.join(OUTPUT_DIR, "00_platform_distribution.csv"), index=False)

# 8. Create y_ck
# y_ck=1 for fine label 2, y_ck=0 for fine label 0, y_ck=-1 for all other classes
df['y_ck'] = df['fine-grained label'].apply(lambda x: 1 if x == 2 else (0 if x == 0 else -1))

# 9. Create locked train/val/test split
# seed=42, ratio=70/10/20
df['stratify_key'] = df['fine-grained label'].astype(str) + "_" + df['topic'].astype(str)

stratification_used = ""
try:
    # First split: train 70%, temp 30%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df['stratify_key']
    )
    # Second split: val 10% of total, test 20% of total.
    # Since temp is 30% of total, val is 1/3, test is 2/3.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=2/3,
        random_state=42,
        stratify=temp_df['stratify_key']
    )
    stratification_used = "fine-grained label x topic"
except ValueError as e:
    # Fallback to fine-grained label
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df['fine-grained label']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=2/3,
        random_state=42,
        stratify=temp_df['fine-grained label']
    )
    stratification_used = "fine-grained label"

# 10. Report CK count per split and compile split distribution report
train_ck_count = (train_df['y_ck'] == 1).sum()
val_ck_count = (val_df['y_ck'] == 1).sum()
test_ck_count = (test_df['y_ck'] == 1).sum()

r_train = train_ck_count / len(train_df)
r_val = val_ck_count / len(val_df)
r_test = test_ck_count / len(test_df)

split_dist = pd.DataFrame([
    {
        'split': 'train',
        'sample_count': len(train_df),
        'percentage': f"{(len(train_df) / total_samples):.4%}",
        'ck_count': train_ck_count,
        'ck_ratio': f"{r_train:.4%}"
    },
    {
        'split': 'val',
        'sample_count': len(val_df),
        'percentage': f"{(len(val_df) / total_samples):.4%}",
        'ck_count': val_ck_count,
        'ck_ratio': f"{r_val:.4%}"
    },
    {
        'split': 'test',
        'sample_count': len(test_df),
        'percentage': f"{(len(test_df) / total_samples):.4%}",
        'ck_count': test_ck_count,
        'ck_ratio': f"{r_test:.4%}"
    }
])
split_dist.to_csv(os.path.join(OUTPUT_DIR, "00_split_distribution.csv"), index=False)

# 11. Add kill-switch warnings
warnings = []

# W1: image missing > 5%
image_missing_rate = 1.0 - image_exists_rate
if image_missing_rate > 0.05:
    warnings.append(f"KILL-SWITCH WARNING: Image missing rate is {image_missing_rate:.4%}, which is above the 5% threshold.")

# W2: knowledge_embedding missing > 10%
if missing_knowledge_embedding_rate > 0.10:
    warnings.append(f"KILL-SWITCH WARNING: Knowledge embedding missing rate is {missing_knowledge_embedding_rate:.4%}, which is above the 10% threshold.")

# W3: CK class in val or test < 30 samples
if val_ck_count < 30:
    warnings.append(f"KILL-SWITCH WARNING: CK class (y_ck=1) in validation split has only {val_ck_count} samples, which is below the threshold of 30.")
if test_ck_count < 30:
    warnings.append(f"KILL-SWITCH WARNING: CK class (y_ck=1) in test split has only {test_ck_count} samples, which is below the threshold of 30.")

# W4: CK ratio train/val/test differs by more than 2x
ratios = [r_train, r_val, r_test]
if min(ratios) == 0:
    ratio_diff = float('inf') if max(ratios) > 0 else 1.0
else:
    ratio_diff = max(ratios) / min(ratios)

if ratio_diff > 2.0:
    warnings.append(f"KILL-SWITCH WARNING: CK ratio train/val/test differs by more than 2x (ratio max/min difference is {ratio_diff:.4f}x).")

# W5: label mapping is unclear
# Check alignment: fine-grained label 0 maps to binary label 1. 1, 2, 3, 4, 5 map to binary label 0.
invalid_binary_labels = set(df['label'].unique()) - {0, 1}
invalid_fine_labels = set(df['fine-grained label'].unique()) - {0, 1, 2, 3, 4, 5}
mismatches = df[((df['fine-grained label'] == 0) & (df['label'] != 1)) |
                ((df['fine-grained label'] > 0) & (df['label'] != 0))]

if len(invalid_binary_labels) > 0 or len(invalid_fine_labels) > 0 or len(mismatches) > 0:
    warnings.append("KILL-SWITCH WARNING: Label mapping is unclear or inconsistent between binary label and fine-grained labels.")

# 12. Save manifests
# Assign splits to df for manifest_all
train_df = train_df.copy()
train_df['split'] = 'train'
val_df = val_df.copy()
val_df['split'] = 'val'
test_df = test_df.copy()
test_df['split'] = 'test'

manifest_all = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Drop helper stratify key
manifest_all_to_save = manifest_all.drop(columns=['stratify_key'])
train_to_save = train_df.drop(columns=['stratify_key', 'split'])
val_to_save = val_df.drop(columns=['stratify_key', 'split'])
test_to_save = test_df.drop(columns=['stratify_key', 'split'])

manifest_all_to_save.to_csv(os.path.join(PROCESSED_DIR, "manifest_all.csv"), index=False)
train_to_save.to_csv(os.path.join(PROCESSED_DIR, "manifest_train_seed42.csv"), index=False)
val_to_save.to_csv(os.path.join(PROCESSED_DIR, "manifest_val_seed42.csv"), index=False)
test_to_save.to_csv(os.path.join(PROCESSED_DIR, "manifest_test_seed42.csv"), index=False)

# Compile summary text
summary_lines = [
    "=================================================================",
    "                    STAGE A DATA AUDIT SUMMARY                   ",
    "=================================================================",
    f"Total Sample Count: {total_samples}",
    f"Total Unique Image Paths in Dataset: {total_unique_image_paths}",
    f"Total Image Files on Disk in Image Directory: {total_physical_images}",
    "",
    "MISSINGNESS REPORT:",
    f"  - Missing Text Rate: {missing_text_rate:.4%} ({missing_text_count}/{total_samples})",
    f"  - Missing Image Path Rate: {missing_image_path_rate:.4%} ({missing_image_path_count}/{total_samples})",
    f"  - Image File Exists Rate: {image_exists_rate:.4%} ({image_exists_count}/{total_samples})",
    f"  - Missing Entity ID Rate: {missing_entity_id_rate:.4%} ({missing_entity_id_count}/{total_samples})",
    f"  - Missing Knowledge Embedding Rate: {missing_knowledge_embedding_rate:.4%} ({missing_knowledge_embedding_count}/{total_samples})",
    f"  - Missing Relation Rate: {missing_relation_rate:.4%} ({missing_relation_count}/{total_samples})",
    "",
    "LABEL DISTRIBUTIONS:",
    "Binary Labels (0=fake, 1=real):",
]
for idx, row in label_dist.iterrows():
    summary_lines.append(f"  - {row['label_name']} ({row['label']}): {row['count']} ({row['percentage']})")

summary_lines.append("")
summary_lines.append("Fine-Grained Labels:")
for idx, row in fine_label_dist.iterrows():
    summary_lines.append(f"  - {row['fine_label_name']} ({row['fine_label']}): {row['count']} ({row['percentage']})")

summary_lines.append("")
summary_lines.append("SPLIT DETAILS:")
summary_lines.append(f"Stratification Used: {stratification_used}")
for idx, row in split_dist.iterrows():
    summary_lines.append(f"  - {row['split'].capitalize()} split: {row['sample_count']} samples ({row['percentage']}) | CK count: {row['ck_count']} | CK ratio: {row['ck_ratio']}")

summary_lines.append("")
summary_lines.append("=================================================================")
if warnings:
    summary_lines.append("KILL-SWITCH WARNINGS DETECTED:")
    for w in warnings:
        summary_lines.append(f"  [TRIGGERED] {w}")
else:
    summary_lines.append("No kill-switch warnings triggered.")
summary_lines.append("=================================================================")

summary_content = "\n".join(summary_lines)

# Write summary file
with open(os.path.join(OUTPUT_DIR, "00_stage_a_summary.txt"), "w") as f:
    f.write(summary_content)

# Output summary to console
print("\n" + summary_content)
print("\nStage A Data Audit completed successfully. All outputs saved.")
