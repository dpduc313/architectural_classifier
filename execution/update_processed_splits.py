"""
update_processed_splits.py — Re-organize processed_data directory and update processed_manifest.csv to reflect the 70-15-15 split.
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, ".tmp", "manifest.csv")
PROC_MANIFEST_PATH = os.path.join(PROJECT_ROOT, ".tmp", "processed_manifest.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")


def main():
    print("Reading manifests...")
    manifest = pd.read_csv(MANIFEST_PATH)
    proc = pd.read_csv(PROC_MANIFEST_PATH)

    # Map file_path -> new split and updated building_id
    split_map = dict(zip(manifest['file_path'], manifest['split']))
    building_map = dict(zip(manifest['file_path'], manifest['building_id']))
    
    proc['new_split'] = proc['file_path'].map(split_map)
    proc['building_id'] = proc['file_path'].map(building_map)

    print("Re-organizing patch files on disk to match 70-15-15 split...")
    moved_count = 0

    for idx, row in tqdm(proc.iterrows(), total=len(proc)):
        old_split = row['split']
        new_split = row['new_split']

        if old_split != new_split:
            style = row['style_label']
            patch_name = os.path.basename(row['processed_path'])

            old_file_path = os.path.join(PROCESSED_DATA_DIR, old_split, style, patch_name)
            new_dir_path  = os.path.join(PROCESSED_DATA_DIR, new_split, style)
            new_file_path = os.path.join(new_dir_path, patch_name)

            os.makedirs(new_dir_path, exist_ok=True)

            if os.path.exists(old_file_path):
                shutil.move(old_file_path, new_file_path)
                moved_count += 1
            elif not os.path.exists(new_file_path):
                print(f"Warning: File not found {old_file_path}")

            proc.at[idx, 'processed_path'] = os.path.join('processed_data', new_split, style, patch_name)

    proc['split'] = proc['new_split']
    proc.drop(columns=['new_split'], inplace=True)
    proc.to_csv(PROC_MANIFEST_PATH, index=False, encoding='utf-8-sig')

    print(f"\nSuccessfully moved {moved_count} patch files.")
    print("\nUpdated processed manifest split summary:")
    print(proc['split'].value_counts())
    print("\nClass distribution per split:")
    print(proc.groupby(['split', 'style_label']).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
