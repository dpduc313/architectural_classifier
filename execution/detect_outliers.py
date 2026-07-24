import os
import torch
import pandas as pd
import numpy as np

manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\manifest.csv"
embeddings_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\embeddings.pt"
outliers_review_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\outliers_review.csv"
outlier_ratio = 0.05

def detect_outliers():
    if not os.path.exists(manifest_path) or not os.path.exists(embeddings_path):
        print("Error: manifest.csv or embeddings.pt not found. Run previous steps first.")
        return

    df = pd.read_csv(manifest_path)
    embeddings = torch.load(embeddings_path)
    n_images = len(df)

    print(f"Loaded {n_images} images and embeddings.")

    norm_embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    df['outlier_score'] = 0.0

    classes = df['style_label'].unique()
    for cls in classes:
        cls_mask = df['style_label'] == cls
        cls_indices = df[cls_mask].index.tolist()
        
        if len(cls_indices) == 0:
            continue
            
        cls_embeddings = norm_embeddings[cls_indices]
        centroid = cls_embeddings.mean(dim=0, keepdim=True)
        centroid = centroid / (centroid.norm(dim=1, keepdim=True) + 1e-8)
        
        similarities = torch.mm(cls_embeddings, centroid.t()).squeeze(1)
        distances = 1.0 - similarities.numpy()
        df.loc[cls_mask, 'outlier_score'] = distances

    n_outliers = int(n_images * outlier_ratio)
    threshold_val = df['outlier_score'].nlargest(n_outliers).iloc[-1]
    df['is_outlier'] = df['outlier_score'] >= threshold_val

    df.to_csv(manifest_path, index=False, encoding='utf-8-sig')

    outliers_df = df[df['is_outlier']].sort_values(by='outlier_score', ascending=False)
    outliers_df.to_csv(outliers_review_path, index=False, encoding='utf-8-sig')
    
    print(f"Outlier detection completed.")
    print(f"  Flagged {len(outliers_df)} images as outliers (Threshold distance: {threshold_val:.4f}).")
    print(f"  Saved manifest with 'is_outlier' flag to: {manifest_path}")
    print(f"  Saved list of outliers for review to: {outliers_review_path}")
    print("\nOutlier count by class:")
    print(outliers_df['style_label'].value_counts())

if __name__ == "__main__":
    detect_outliers()
