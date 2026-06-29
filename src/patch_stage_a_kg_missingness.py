import os
import sys
import ast
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
WORKSPACE_DIR = r"D:\CIKD"
PROCESSED_DIR = os.path.join(WORKSPACE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs", "stage_a_audit")

# Ensure output directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Input and Output Paths
# ---------------------------------------------------------
MANIFEST_PATHS = {
    'all': os.path.join(PROCESSED_DIR, "manifest_all.csv"),
    'train': os.path.join(PROCESSED_DIR, "manifest_train_seed42.csv"),
    'val': os.path.join(PROCESSED_DIR, "manifest_val_seed42.csv"),
    'test': os.path.join(PROCESSED_DIR, "manifest_test_seed42.csv")
}

KG_COMPLETE_MANIFEST_PATHS = {
    'all': os.path.join(PROCESSED_DIR, "manifest_kg_complete_all.csv"),
    'train': os.path.join(PROCESSED_DIR, "manifest_kg_complete_train_seed42.csv"),
    'val': os.path.join(PROCESSED_DIR, "manifest_kg_complete_val_seed42.csv"),
    'test': os.path.join(PROCESSED_DIR, "manifest_kg_complete_test_seed42.csv")
}

TVCS_ELIGIBLE_MANIFEST_PATHS = {
    'all': os.path.join(PROCESSED_DIR, "manifest_tvcs_eligible_all.csv"),
    'train': os.path.join(PROCESSED_DIR, "manifest_tvcs_eligible_train_seed42.csv"),
    'val': os.path.join(PROCESSED_DIR, "manifest_tvcs_eligible_val_seed42.csv"),
    'test': os.path.join(PROCESSED_DIR, "manifest_tvcs_eligible_test_seed42.csv")
}

REPORT_PATHS = {
    'kg_missing_by_fine_label': os.path.join(OUTPUT_DIR, "00_kg_missing_by_fine_label_kg_patch.csv"),
    'kg_complete_split_dist': os.path.join(OUTPUT_DIR, "00_kg_complete_split_distribution_kg_patch.csv"),
    'tvcs_eligible_dist': os.path.join(OUTPUT_DIR, "00_tvcs_eligible_distribution_kg_patch.csv"),
    'summary_txt': os.path.join(OUTPUT_DIR, "00_stage_a_kg_patch_summary.txt")
}

# ---------------------------------------------------------
# Labels Mapping
# ---------------------------------------------------------
BINARY_LABEL_MAP = {0: 'fake', 1: 'real'}
FINE_LABEL_MAP = {
    0: 'real',
    1: 'text-image inconsistency',
    2: 'content-knowledge inconsistency',
    3: 'text-based fake',
    4: 'image-based fake',
    5: 'others'
}

# ---------------------------------------------------------
# Helper functions for checking missingness
# ---------------------------------------------------------
def check_has_knowledge_embedding(val):
    if pd.isnull(val):
        return False
    val_str = str(val).strip()
    if val_str == "" or val_str == "[]" or val_str == "None":
        return False
    cleaned = val_str.replace('[', '').replace(']', '').replace('\n', ' ').strip()
    if not cleaned:
        return False
    try:
        parts = [float(x) for x in cleaned.split() if x.strip()]
        if len(parts) == 0:
            return False
        return not all(x == 0.0 for x in parts)
    except ValueError:
        return False

def check_has_list(val):
    if pd.isnull(val):
        return False
    val_str = str(val).strip()
    if val_str == "" or val_str == "[]" or val_str == "None":
        return False
    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            return len(parsed) > 0
        return True
    except Exception:
        # Fallback if parsing fails but string is non-empty and has content
        return len(val_str) > 2

# ---------------------------------------------------------
# Processing logic
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("Running Stage A KG Missingness Patch...")
    print("=" * 60)

    # Check existence of input manifest files
    for name, path in MANIFEST_PATHS.items():
        if not os.path.exists(path):
            print(f"ERROR: Input manifest for '{name}' not found at: {path}")
            sys.exit(1)

    # Load all manifests
    dfs = {}
    for name, path in MANIFEST_PATHS.items():
        print(f"Loading {name} manifest from {path}...")
        dfs[name] = pd.read_csv(path)
        print(f"  Loaded {len(dfs[name])} rows.")

    # 1. Add Columns to DataFrames
    for name in dfs:
        df = dfs[name]
        print(f"\nProcessing columns for '{name}' manifest...")
        
        # Add fine_label column if not present (as alias for fine-grained label)
        if 'fine_label' not in df.columns:
            if 'fine-grained label' in df.columns:
                df['fine_label'] = df['fine-grained label']
            else:
                print("ERROR: Neither 'fine_label' nor 'fine-grained label' found in manifest columns.")
                sys.exit(1)
        
        # Compute component missingness
        df['has_knowledge_embedding'] = df['knowledge_embedding'].apply(check_has_knowledge_embedding)
        df['has_relation'] = df['relation'].apply(check_has_list)
        df['has_entity_id'] = df['entity_id'].apply(check_has_list)
        
        # Compute complete and eligible flags
        df['kg_complete'] = df['has_knowledge_embedding'] & df['has_relation']
        df['tvcs_eligible'] = df['kg_complete'] & df['fine_label'].isin([0, 2])
        
        # Save back to original file path with added columns
        target_path = MANIFEST_PATHS[name]
        df.to_csv(target_path, index=False)
        print(f"  Saved updated {name} manifest to {target_path}")

    # Reference manifest_all for full statistics
    df_all = dfs['all']

    # 2. Filter and create KG-complete manifests
    for name, path in KG_COMPLETE_MANIFEST_PATHS.items():
        df_filtered = dfs[name][dfs[name]['kg_complete'] == True].copy()
        df_filtered.to_csv(path, index=False)
        print(f"Created KG-complete manifest '{name}' with {len(df_filtered)} rows at: {path}")

    # 3. Filter and create TVCS-eligible manifests
    for name, path in TVCS_ELIGIBLE_MANIFEST_PATHS.items():
        df_filtered = dfs[name][dfs[name]['tvcs_eligible'] == True].copy()
        df_filtered.to_csv(path, index=False)
        print(f"Created TVCS-eligible manifest '{name}' with {len(df_filtered)} rows at: {path}")

    # 4. Generate Reports
    # Report A: KG missingness by fine-grained label
    # Group by fine_label and count missing rates
    fine_label_groups = []
    for label_val, label_name in FINE_LABEL_MAP.items():
        sub_df = df_all[df_all['fine_label'] == label_val]
        total_count = len(sub_df)
        if total_count > 0:
            missing_embed = total_count - sub_df['has_knowledge_embedding'].sum()
            missing_rel = total_count - sub_df['has_relation'].sum()
            missing_ent = total_count - sub_df['has_entity_id'].sum()
            kg_complete_cnt = sub_df['kg_complete'].sum()
            
            fine_label_groups.append({
                'fine_label': label_val,
                'fine_label_name': label_name,
                'total_count': total_count,
                'missing_knowledge_embedding_count': missing_embed,
                'missing_knowledge_embedding_rate': f"{(missing_embed / total_count):.4%}",
                'missing_relation_count': missing_rel,
                'missing_relation_rate': f"{(missing_rel / total_count):.4%}",
                'missing_entity_id_count': missing_ent,
                'missing_entity_id_rate': f"{(missing_ent / total_count):.4%}",
                'kg_complete_count': kg_complete_cnt,
                'kg_complete_rate': f"{(kg_complete_cnt / total_count):.4%}"
            })
    fine_label_df = pd.DataFrame(fine_label_groups)
    fine_label_df.to_csv(REPORT_PATHS['kg_missing_by_fine_label'], index=False)
    print(f"\nSaved KG missingness by fine label report to: {REPORT_PATHS['kg_missing_by_fine_label']}")

    # Report B: KG-complete split distribution
    split_dist_rows = []
    for split_name in ['train', 'val', 'test']:
        split_df = dfs[split_name]
        total_count = len(split_df)
        kg_complete_cnt = split_df['kg_complete'].sum()
        # CK class is fine_label == 2 (content-knowledge inconsistency)
        kg_complete_ck_cnt = ((split_df['kg_complete'] == True) & (split_df['fine_label'] == 2)).sum()
        
        split_dist_rows.append({
            'split': split_name,
            'total_samples': total_count,
            'kg_complete_count': kg_complete_cnt,
            'kg_complete_rate': f"{(kg_complete_cnt / total_count):.4%}",
            'kg_complete_ck_count': kg_complete_ck_cnt,
            'kg_complete_ck_rate': f"{(kg_complete_ck_cnt / total_count):.4%}"
        })
    split_dist_df = pd.DataFrame(split_dist_rows)
    split_dist_df.to_csv(REPORT_PATHS['kg_complete_split_dist'], index=False)
    print(f"Saved KG-complete split distribution report to: {REPORT_PATHS['kg_complete_split_dist']}")

    # Report C: TVCS-eligible distribution
    tvcs_dist_rows = []
    for split_name in ['train', 'val', 'test']:
        split_df = dfs[split_name]
        total_count = len(split_df)
        tvcs_eligible_cnt = split_df['tvcs_eligible'].sum()
        
        tvcs_dist_rows.append({
            'split': split_name,
            'total_samples': total_count,
            'tvcs_eligible_count': tvcs_eligible_cnt,
            'tvcs_eligible_rate': f"{(tvcs_eligible_cnt / total_count):.4%}"
        })
    tvcs_dist_df = pd.DataFrame(tvcs_dist_rows)
    tvcs_dist_df.to_csv(REPORT_PATHS['tvcs_eligible_dist'], index=False)
    print(f"Saved TVCS-eligible distribution report to: {REPORT_PATHS['tvcs_eligible_dist']}")

    # Decision Logic
    # If KG-complete CK count in val/test >= 30, CIKD/TVCS is safe.
    val_kg_complete_ck = split_dist_df.loc[split_dist_df['split'] == 'val', 'kg_complete_ck_count'].values[0]
    test_kg_complete_ck = split_dist_df.loc[split_dist_df['split'] == 'test', 'kg_complete_ck_count'].values[0]
    
    is_safe = (val_kg_complete_ck >= 30) and (test_kg_complete_ck >= 30)
    
    decision_text = ""
    if is_safe:
        decision_text = (
            f"CIKD/TVCS IS SAFE.\n"
            f"Reason: KG-complete CK count in validation ({val_kg_complete_ck}) and test ({test_kg_complete_ck}) splits are both >= 30."
        )
    else:
        decision_text = (
            f"CIKD/TVCS IS NOT SAFE (KG-complete CK count is low).\n"
            f"Action: Train CIKD with kg_available mask and weaken TVCS claim.\n"
            f"Reason: KG-complete CK count in validation ({val_kg_complete_ck}) and/or test ({test_kg_complete_ck}) split is < 30."
        )

    # 5. Compile Text Summary Report
    summary_lines = [
        "=" * 65,
        "                    STAGE A KG PATCH SUMMARY REPORT              ",
        "=" * 65,
        f"Total Samples Analyzed: {len(df_all)}",
        "",
        "OVERALL KG MISSINGNESS (All Splits):",
        f"  - Missing Knowledge Embedding: {((len(df_all) - df_all['has_knowledge_embedding'].sum()) / len(df_all)):.4%} ({len(df_all) - df_all['has_knowledge_embedding'].sum()}/{len(df_all)})",
        f"  - Missing Relation:            {((len(df_all) - df_all['has_relation'].sum()) / len(df_all)):.4%} ({len(df_all) - df_all['has_relation'].sum()}/{len(df_all)})",
        f"  - Missing Entity ID:           {((len(df_all) - df_all['has_entity_id'].sum()) / len(df_all)):.4%} ({len(df_all) - df_all['has_entity_id'].sum()}/{len(df_all)})",
        f"  - KG Incomplete Rate:          {((len(df_all) - df_all['kg_complete'].sum()) / len(df_all)):.4%} ({len(df_all) - df_all['kg_complete'].sum()}/{len(df_all)})",
        f"  - KG Complete Rate:            {(df_all['kg_complete'].sum() / len(df_all)):.4%} ({df_all['kg_complete'].sum()}/{len(df_all)})",
        "",
        "-" * 65,
        "MISSINGNESS REPORT BY GROUPINGS",
        "-" * 65,
    ]

    # Helper function to generate pretty text tables
    def build_group_summary(group_col, group_name_map=None):
        table_lines = []
        # Header
        table_lines.append(f"KG Missingness by {group_col.replace('_', ' ').title()}:")
        table_lines.append(f"{'Group Value':<30} | {'Total':<8} | {'Embed%':<8} | {'Rel%':<8} | {'EntID%':<8} | {'KG Compl%':<10}")
        table_lines.append("-" * 80)
        
        unique_vals = sorted(df_all[group_col].dropna().unique())
        for val in unique_vals:
            sub = df_all[df_all[group_col] == val]
            cnt = len(sub)
            if cnt == 0:
                continue
            emb_missing_pct = (1.0 - sub['has_knowledge_embedding'].mean()) * 100
            rel_missing_pct = (1.0 - sub['has_relation'].mean()) * 100
            ent_missing_pct = (1.0 - sub['has_entity_id'].mean()) * 100
            kg_compl_pct = sub['kg_complete'].mean() * 100
            
            lbl_name = str(val)
            if group_name_map and val in group_name_map:
                lbl_name = f"{val} ({group_name_map[val]})"
            
            table_lines.append(f"{lbl_name:<30} | {cnt:<8} | {emb_missing_pct:>7.2f}% | {rel_missing_pct:>7.2f}% | {ent_missing_pct:>7.2f}% | {kg_compl_pct:>9.2f}%")
        return "\n".join(table_lines) + "\n"

    # Add groupings
    summary_lines.append(build_group_summary('fine_label', FINE_LABEL_MAP))
    summary_lines.append(build_group_summary('label', BINARY_LABEL_MAP))
    summary_lines.append(build_group_summary('topic'))
    summary_lines.append(build_group_summary('platform'))
    summary_lines.append(build_group_summary('split'))

    summary_lines.extend([
        "-" * 65,
        "CIKD/TVCS SAFETY DECISION AND ACTION PLAN",
        "-" * 65,
        f"Validation Split KG-complete CK sample count: {val_kg_complete_ck}",
        f"Test Split KG-complete CK sample count:       {test_kg_complete_ck}",
        "",
        "DECISION RESULT:",
        decision_text,
        "=" * 65,
    ])

    summary_content = "\n".join(summary_lines)
    with open(REPORT_PATHS['summary_txt'], 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print("\nStage A KG Missingness Patch completed successfully!")
    print("\nSummary Report Output:")
    print(summary_content)

if __name__ == "__main__":
    main()
