import os
import torch
import pandas as pd
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest.csv"
CLEANED_MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest_cleaned.csv"
CLEANED_DIR = PROJECT_ROOT / "processed_data_cleaned"

BATCH_SIZE = 128
CONF_THRESHOLD = 0.25
BUILDING_THRESHOLD = 0.018  # Keep if building covers >= 1.8% of patch (retains ~120,000 patches)

def main():
    # 1. Clean up old runs to start fresh with new threshold
    if CLEANED_MANIFEST_PATH.exists():
        os.remove(CLEANED_MANIFEST_PATH)
    if CLEANED_DIR.exists():
        import shutil
        shutil.rmtree(CLEANED_DIR)

    # 2. Download/Load model
    print("Loading YOLOv8 building segmentation model...")
    model_path = hf_hub_download(repo_id="keremberke/yolov8m-building-segmentation", filename="best.pt")
    model = YOLO(model_path)
    
    # 2. Read processed manifest
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return
        
    df = pd.read_csv(MANIFEST_PATH)
    total_patches = len(df)
    print(f"Loaded manifest with {total_patches:,} patches.")
    
    # 3. Check for existing progress (resumability)
    processed_paths = set()
    cleaned_rows = []
    
    if CLEANED_MANIFEST_PATH.exists():
        try:
            df_existing = pd.read_csv(CLEANED_MANIFEST_PATH)
            processed_paths = set(df_existing['processed_path'].tolist())
            cleaned_rows = df_existing.to_dict('records')
            print(f"Found existing progress: {len(processed_paths):,} patches already processed.")
        except Exception as e:
            print(f"Warning: Could not read existing cleaned manifest: {e}. Starting fresh.")
            
    # Filter out already processed paths
    pending_df = df[~df['processed_path'].isin(processed_paths)].copy()
    num_pending = len(pending_df)
    print(f"Pending patches to process: {num_pending:,}")
    
    if num_pending == 0:
        print("All patches already processed!")
        return

    # Create generator for batch processing
    pending_records = pending_df.to_dict('records')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on device: {device} (Batch size: {BATCH_SIZE})")
    
    # Process in batches
    for i in tqdm(range(0, num_pending, BATCH_SIZE), desc="Processing Batches"):
        batch_records = pending_records[i : i + BATCH_SIZE]
        
        # Resolve absolute paths of patches
        batch_absolute_paths = []
        for r in batch_records:
            abs_path = PROJECT_ROOT / r['processed_path']
            batch_absolute_paths.append(str(abs_path))
            
        # Run prediction
        try:
            results = model.predict(
                source=batch_absolute_paths,
                device=device,
                conf=CONF_THRESHOLD,
                verbose=False
            )
        except Exception as e:
            print(f"\nError running prediction on batch starting at index {i}: {e}")
            continue
            
        # Analyze results
        for record, result in zip(batch_records, results):
            if result.masks is not None and len(result.masks.data) > 0:
                combined_mask = torch.any(result.masks.data > 0.5, dim=0)
                building_ratio = combined_mask.float().mean().item()
            else:
                building_ratio = 0.0
                
            # If building covers >= 70% of patch, we keep it
            if building_ratio >= BUILDING_THRESHOLD:
                # Target path in cleaned directory
                original_rel_path = record['processed_path'] # processed_data/split/style_label/filename
                cleaned_rel_path = original_rel_path.replace("processed_data", "processed_data_cleaned")
                
                src_path = PROJECT_ROOT / original_rel_path
                dst_path = PROJECT_ROOT / cleaned_rel_path
                
                # Create destination directory
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Link the file (hard link)
                if not dst_path.exists():
                    try:
                        os.link(src_path, dst_path)
                    except Exception as e:
                        # Fallback to copy if hard link fails
                        import shutil
                        try:
                            shutil.copy(src_path, dst_path)
                        except Exception as copy_err:
                            print(f"\nFailed to link/copy {src_path} to {dst_path}: {copy_err}")
                            continue
                
                # Record in cleaned list
                new_record = record.copy()
                new_record['processed_path'] = cleaned_rel_path
                new_record['building_ratio'] = round(building_ratio, 4)
                cleaned_rows.append(new_record)
            
            processed_paths.add(record['processed_path'])
            
        # Periodically save progress every 5 batches (640 images) to prevent data loss
        if i % (BATCH_SIZE * 5) == 0 or (i + BATCH_SIZE) >= num_pending:
            temp_df = pd.DataFrame(cleaned_rows)
            temp_df.to_csv(CLEANED_MANIFEST_PATH, index=False, encoding='utf-8-sig')

    print(f"\nFiltering complete! Cleaned dataset contains {len(cleaned_rows):,} / {total_patches:,} patches.")
    print(f"Cleaned manifest saved to: {CLEANED_MANIFEST_PATH}")
    print(f"Cleaned images are located in: {CLEANED_DIR}")

if __name__ == "__main__":
    main()
