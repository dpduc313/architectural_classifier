import os
import shutil
import pandas as pd
from PIL import Image, ImageStat
import numpy as np
from tqdm import tqdm

manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\manifest.csv"
project_root = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final"
output_dir = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\processed_data"
processed_manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\processed_manifest.csv"
patch_size = 1000
variance_threshold = 150.0  # Threshold for dropping empty/sky patches

def is_low_variance(patch_img):
    """Returns True if the patch is low variance (e.g., flat sky, blank wall)"""
    gray = patch_img.convert("L")
    stat = ImageStat.Stat(gray)
    return stat.var[0] < variance_threshold

def preprocess_images():
    if not os.path.exists(manifest_path):
        print("Error: manifest.csv not found. Run previous steps first.")
        return

    df = pd.read_csv(manifest_path)
    
    # We will assume detect_outliers.py and split_dataset.py have added 'is_outlier', 'split', and 'dup_cluster_id'
    if 'is_outlier' not in df.columns:
        print("Warning: 'is_outlier' column missing. Did you run detect_outliers.py?")
        df['is_outlier'] = False
    if 'split' not in df.columns:
        print("Warning: 'split' column missing. Did you run split_dataset.py?")
        df['split'] = 'train'
    if 'dup_cluster_id' not in df.columns:
        print("Warning: 'dup_cluster_id' missing. Did you run detect_duplicates.py?")
        df['dup_cluster_id'] = -1

    valid_df = df[~df['is_outlier']].copy()
    print(f"Images remaining after removing outliers: {len(valid_df)}")
    
    keep_indices = []
    
    # Deduplication Policy
    # Process train split
    train_split = valid_df[valid_df['split'] == 'train']
    for cluster_id, group in train_split.groupby('dup_cluster_id'):
        if cluster_id == -1:
            keep_indices.extend(group.index.tolist())
        else:
            keep_indices.extend(group.index[:2].tolist())
            
    # Process val split
    val_split = valid_df[valid_df['split'] == 'val']
    for cluster_id, group in val_split.groupby('dup_cluster_id'):
        if cluster_id == -1:
            keep_indices.extend(group.index.tolist())
        else:
            keep_indices.extend(group.index[:1].tolist())
            
    # Process test split
    test_split = valid_df[valid_df['split'] == 'test']
    for cluster_id, group in test_split.groupby('dup_cluster_id'):
        if cluster_id == -1:
            keep_indices.extend(group.index.tolist())
        else:
            keep_indices.extend(group.index[:1].tolist())

    processed_df = valid_df.loc[keep_indices].copy()
    print(f"Original images to process after deduplication: {len(processed_df)}")

    if os.path.exists(output_dir):
        print(f"Cleaning existing processed data directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    patch_records = []
    empty_patches_dropped = 0

    print(f"Extracting {patch_size}x{patch_size} patches...")
    for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df)):
        src_path = os.path.join(project_root, row['file_path'])
        
        dst_subdir = os.path.join(output_dir, row['split'], row['style_label'])
        os.makedirs(dst_subdir, exist_ok=True)
        
        try:
            with Image.open(src_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                
                # Calculate number of patches in each dimension (dropping remainders)
                x_patches = width // patch_size
                y_patches = height // patch_size
                
                for y in range(y_patches):
                    for x in range(x_patches):
                        left = x * patch_size
                        upper = y * patch_size
                        right = left + patch_size
                        lower = upper + patch_size
                        
                        patch = img.crop((left, upper, right, lower))
                        
                        if is_low_variance(patch):
                            empty_patches_dropped += 1
                            continue
                            
                        # Save patch
                        filename_no_ext, ext = os.path.splitext(row['filename'])
                        patch_filename = f"{filename_no_ext}_patch_{y}_{x}.jpg"
                        patch_dst_path = os.path.join(dst_subdir, patch_filename)
                        
                        patch.save(patch_dst_path, 'JPEG', quality=95)
                        
                        # Record patch metadata
                        patch_record = row.copy()
                        patch_record['original_filename'] = row['filename']
                        patch_record['filename'] = patch_filename
                        patch_record['patch_y'] = y
                        patch_record['patch_x'] = x
                        patch_record['processed_path'] = f"processed_data/{row['split']}/{row['style_label']}/{patch_filename}"
                        patch_records.append(patch_record)
                        
        except Exception as e:
            print(f"\nError processing image {src_path}: {e}")

    print(f"\nPreprocessing finished!")
    print(f"Successfully generated {len(patch_records)} high-quality patches.")
    print(f"Dropped {empty_patches_dropped} low-variance/empty patches.")
    
    final_df = pd.DataFrame(patch_records)
    final_df.to_csv(processed_manifest_path, index=False, encoding='utf-8-sig')
    print(f"Saved processed manifest to: {processed_manifest_path}")

if __name__ == "__main__":
    preprocess_images()
