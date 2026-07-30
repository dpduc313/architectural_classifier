import os
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest.csv"

def main():
    # 1. Load model
    print("Loading YOLOv8 building segmentation model...")
    model_path = hf_hub_download(repo_id="keremberke/yolov8m-building-segmentation", filename="best.pt")
    model = YOLO(model_path)
    
    # 2. Read manifest and sample 1000 patches
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return
        
    df = pd.read_csv(MANIFEST_PATH)
    sample_df = df.sample(n=1000, random_state=42)
    print(f"Sampled 1,000 patches for analysis.")
    
    absolute_paths = [str(PROJECT_ROOT / r['processed_path']) for r in sample_df.to_dict('records')]
    
    # 3. Predict on sample in batches
    ratios = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Batch size 128
    batch_size = 128
    for i in range(0, len(absolute_paths), batch_size):
        batch = absolute_paths[i:i+batch_size]
        results = model.predict(source=batch, device=device, conf=0.15, verbose=False) # lowered conf slightly to detect lower confidence buildings
        for res in results:
            if res.masks is not None and len(res.masks.data) > 0:
                combined_mask = torch.any(res.masks.data > 0.5, dim=0)
                ratios.append(combined_mask.float().mean().item())
            else:
                ratios.append(0.0)
                
    ratios = np.array(ratios)
    
    # 4. Print percentiles
    print("\n--- Percentiles of Building Coverage Ratio (1,000 Sample Patches) ---")
    for p in range(10, 101, 10):
        val = np.percentile(ratios, p)
        print(f"{p}th percentile: {val:.4f} (keeping {100 - p}% of data)")
        
    # Find threshold for keeping ~65.3% of data (which is 34.7th percentile)
    thresh_65 = np.percentile(ratios, 34.7)
    print(f"\nTo keep 65.3% of the dataset (approx 120,000 patches), the threshold should be: {thresh_65:.4f}")

if __name__ == "__main__":
    main()
