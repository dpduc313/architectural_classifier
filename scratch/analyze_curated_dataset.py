import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_CLEANED_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest_cleaned.csv"

def get_subpath(p):
    p_str = str(p).replace('\\', '/')
    for split in ['train/', 'val/', 'test/']:
        if split in p_str:
            return p_str.split(split, 1)[1]
    return p_str

def main():
    if not MANIFEST_CLEANED_PATH.exists():
        print("Cleaned manifest not found.")
        return

    print("Loading original clean manifest (YOLOv8 >= 1.8%)...")
    df_cleaned = pd.read_csv(MANIFEST_CLEANED_PATH)
    original_count = len(df_cleaned)
    
    # Store set of currently kept subpaths
    cleaned_subpaths = set(df_cleaned['processed_path'].apply(get_subpath).tolist())

    # Find all CSV files
    all_files = os.listdir(PROJECT_ROOT)
    move_to_kept_files = [f for f in all_files if f.startswith("move_to_kept_patches") and f.endswith(".csv")]
    filter_out_files = [f for f in all_files if f.startswith("filter_out_patches") and f.endswith(".csv")]

    print(f"Found {len(move_to_kept_files)} addition CSV files and {len(filter_out_files)} removal CSV files.")

    # 1. Process additions (move_to_kept_patches)
    added_subpaths = set()
    for f_name in move_to_kept_files:
        try:
            df = pd.read_csv(PROJECT_ROOT / f_name)
            for _, row in df.iterrows():
                if row['new_action'] == 'KEEP':
                    added_subpaths.add(get_subpath(row['processed_path']))
        except Exception as e:
            print(f"Error reading {f_name}: {e}")

    # 2. Process removals (filter_out_patches)
    removed_subpaths = set()
    for f_name in filter_out_files:
        try:
            df = pd.read_csv(PROJECT_ROOT / f_name)
            for _, row in df.iterrows():
                if row['new_action'] == 'FILTER':
                    removed_subpaths.add(get_subpath(row['processed_path']))
        except Exception as e:
            print(f"Error reading {f_name}: {e}")

    print(f"Total unique patches marked to ADD: {len(added_subpaths):,}")
    print(f"Total unique patches marked to REMOVE: {len(removed_subpaths):,}")

    # Calculate new set of kept subpaths
    # New Kept = (Original Cleaned - Removals) + Additions
    final_subpaths = (cleaned_subpaths - removed_subpaths) | added_subpaths

    # Read original full manifest to reconstruct styling info
    df_all = pd.read_csv(PROJECT_ROOT / ".tmp" / "processed_manifest.csv")
    df_all['subpath'] = df_all['processed_path'].apply(get_subpath)
    
    df_final = df_all[df_all['subpath'].isin(final_subpaths)].copy()

    print("\n" + "="*50)
    print("         DATASET CURATION REPORT (PHASE 2)")
    print("="*50)
    print(f"Original YOLOv8 >=1.8% Dataset:   {original_count:,} patches")
    print(f"User Added (mistakenly filtered):  +{len(added_subpaths):,} patches")
    print(f"User Removed (still noisy/sky):   -{len(removed_subpaths):,} patches")
    print(f"Final Curated Dataset Size:        {len(df_final):,} patches")
    print(f"Net change:                        {len(df_final) - original_count:+,} patches")
    print("-"*50)
    
    print("\nClass Distribution of Curated Dataset:")
    class_counts = df_final['style_label'].value_counts()
    class_pcts = df_final['style_label'].value_counts(normalize=True) * 100
    for cls in sorted(class_counts.index):
        print(f"  * {cls}: {class_counts[cls]:,} patches ({class_pcts[cls]:.2f}%)")

    print("\nDataset Split Distribution:")
    split_counts = df_final['split'].value_counts()
    split_pcts = df_final['split'].value_counts(normalize=True) * 100
    for spl in ['train', 'val', 'test']:
        if spl in split_counts:
            print(f"  * {spl.capitalize()}: {split_counts[spl]:,} patches ({split_pcts[spl]:.2f}%)")

if __name__ == "__main__":
    main()
