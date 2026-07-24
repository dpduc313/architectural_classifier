import os
import pandas as pd
import numpy as np

manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\manifest.csv"
random_seed = 42

def split_dataset():
    if not os.path.exists(manifest_path):
        print("Error: manifest.csv not found. Run previous steps first.")
        return

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} images.")

    # Identify forced test buildings (not standardized or old pics)
    forced_test_mask = (
        df['file_path'].str.contains('HistoricVietnam-OldPics', case=False, na=False) |
        df['file_path'].str.contains('Other-Modernism', case=False, na=False)
    )
    
    # We assign split based on building_id to prevent leakage
    forced_test_buildings = set(df[forced_test_mask]['building_id'].unique())
    print(f"Forced to test split: {len(forced_test_buildings)} buildings based on custom rules.")

    building_styles = df.groupby('building_id')['style_label'].first().reset_index()
    
    np.random.seed(random_seed)
    building_to_split = {}
    
    for style, group in building_styles.groupby('style_label'):
        buildings = group['building_id'].values
        
        b_forced = [b for b in buildings if b in forced_test_buildings]
        b_remaining = [b for b in buildings if b not in forced_test_buildings]
        
        np.random.shuffle(b_remaining)
        
        n_remaining = len(b_remaining)
        n_train = max(1, int(round(n_remaining * 0.70)))
        n_val   = max(1, int(round(n_remaining * 0.15)))
            
        train_b = b_remaining[:n_train]
        val_b   = b_remaining[n_train:n_train + n_val]
        test_b  = b_remaining[n_train + n_val:] + b_forced
        
        for b in train_b:
            building_to_split[b] = 'train'
        for b in val_b:
            building_to_split[b] = 'val'
        for b in test_b:
            building_to_split[b] = 'test'

    df['split'] = df['building_id'].map(building_to_split)
    
    df.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    print("Dataset split completed successfully.")
    print("\nImage counts per split:")
    print(df['split'].value_counts())
    print("\nBuilding counts per split:")
    print(df.groupby('split')['building_id'].nunique())
    print("\nClass distribution per split (images):")
    print(df.groupby(['split', 'style_label']).size().unstack(fill_value=0))

if __name__ == "__main__":
    split_dataset()
