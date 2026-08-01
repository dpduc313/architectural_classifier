import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest.csv"
MANIFEST_CURATED_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest_curated.csv"

def get_subpath(p):
    p_str = str(p).replace('\\', '/')
    for split in ['train/', 'val/', 'test/']:
        if split in p_str:
            return p_str.split(split, 1)[1]
    return p_str

def main():
    if not MANIFEST_PATH.exists() or not MANIFEST_CURATED_PATH.exists():
        print("Error: Required manifest files not found.")
        return

    print("Loading manifests...")
    df_all = pd.read_csv(MANIFEST_PATH)
    df_curated = pd.read_csv(MANIFEST_CURATED_PATH)

    curated_subpaths = set(df_curated['processed_path'].apply(get_subpath).tolist())
    df_all['subpath'] = df_all['processed_path'].apply(get_subpath)
    
    # Attach sub-label: 1 for Architectural (Kept), 0 for Non-Architectural (Filtered out)
    df_all['architectural_sublabel'] = df_all['subpath'].isin(curated_subpaths).astype(int)

    # Classify folders as Standardized vs Non-Standardized
    def is_standardized_path(file_path):
        fp = str(file_path)
        # Exclude known noisy / non-standardized patterns
        if any(x in fp for x in ['HistoricVietnam-OldPics', 'Hanoi', 'need_review', 'Other-Modernism']):
            return False
        # Check standard pattern like A1.01, B2.14
        match = re.search(r'\b[AB][12]\.\d+\b', fp)
        return match is not None

    df_all['is_standardized'] = df_all['file_path'].apply(is_standardized_path)

    df_std = df_all[df_all['is_standardized']].copy()
    df_ref = df_all[~df_all['is_standardized']].copy()

    print(f"Total Patches: {len(df_all):,}")
    print(f"  - Standardized Patches: {len(df_std):,}")
    print(f"  - Non-Standardized (Reference) Patches: {len(df_ref):,}")
    print(f"Architectural Sub-label Distribution (Total):")
    print(df_all['architectural_sublabel'].value_counts())

    # Stratified Split by building_id for Standardized dataset
    np.random.seed(42)
    building_styles = df_std.groupby('building_id')['style_label'].first().reset_index()
    building_to_split = {}

    for style, group in building_styles.groupby('style_label'):
        buildings = group['building_id'].values
        np.random.shuffle(buildings)
        n = len(buildings)
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15)))

        train_b = buildings[:n_train]
        val_b = buildings[n_train:n_train + n_val]
        test_b = buildings[n_train + n_val:]

        for b in train_b:
            building_to_split[b] = 'train'
        for b in val_b:
            building_to_split[b] = 'val'
        for b in test_b:
            building_to_split[b] = 'test'

    df_std['final_split'] = df_std['building_id'].map(building_to_split)
    df_ref['final_split'] = 'reference_test'

    df_train = df_std[df_std['final_split'] == 'train'].copy()
    df_val = df_std[df_std['final_split'] == 'val'].copy()
    df_test = df_std[df_std['final_split'] == 'test'].copy()

    # Save final split manifests
    df_train.to_csv(PROJECT_ROOT / ".tmp" / "final_train_manifest.csv", index=False)
    df_val.to_csv(PROJECT_ROOT / ".tmp" / "final_val_manifest.csv", index=False)
    df_test.to_csv(PROJECT_ROOT / ".tmp" / "final_test_manifest.csv", index=False)
    df_ref.to_csv(PROJECT_ROOT / ".tmp" / "final_reference_test_manifest.csv", index=False)

    print("\nFinal Manifest Splits Saved:")
    print(f"  - Train: {len(df_train):,} patches")
    print(f"  - Val:   {len(df_val):,} patches")
    print(f"  - Test:  {len(df_test):,} patches")
    print(f"  - Ref:   {len(df_ref):,} patches")

if __name__ == "__main__":
    main()
