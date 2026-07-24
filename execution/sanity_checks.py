import os
import pandas as pd

processed_manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\processed_manifest.csv"

def run_sanity_checks():
    if not os.path.exists(processed_manifest_path):
        print(f"Error: Processed manifest not found at {processed_manifest_path}. Run preprocess_images.py first.")
        return

    df = pd.read_csv(processed_manifest_path)
    print("==================================================")
    print("             DATASET SANITY CHECKS                ")
    print("==================================================")

    train_buildings = set(df[df['split'] == 'train']['building_id'].unique())
    val_buildings = set(df[df['split'] == 'val']['building_id'].unique())
    test_buildings = set(df[df['split'] == 'test']['building_id'].unique())

    leakage_train_val = train_buildings.intersection(val_buildings)
    leakage_train_test = train_buildings.intersection(test_buildings)
    leakage_val_test = val_buildings.intersection(test_buildings)

    print("--- 1. BUILDING LEAKAGE CHECK ---")
    leak_detected = False
    if leakage_train_val:
        print(f"  [FAIL] Leakage between Train and Val: {len(leakage_train_val)} buildings overlap! {list(leakage_train_val)[:5]}...")
        leak_detected = True
    if leakage_train_test:
        print(f"  [FAIL] Leakage between Train and Test: {len(leakage_train_test)} buildings overlap! {list(leakage_train_test)[:5]}...")
        leak_detected = True
    if leakage_val_test:
        print(f"  [FAIL] Leakage between Val and Test: {len(leakage_val_test)} buildings overlap! {list(leakage_val_test)[:5]}...")
        leak_detected = True

    if not leak_detected:
        print("  [PASS] No building leakage found between Train, Val, and Test splits.")
        print(f"    Train unique buildings: {len(train_buildings)}")
        print(f"    Val unique buildings: {len(val_buildings)}")
        print(f"    Test unique buildings: {len(test_buildings)}")

    print("\n--- 2. IMAGE COUNT CLASS BALANCE PER SPLIT ---")
    class_balance = df.groupby(['split', 'style_label']).size().unstack(fill_value=0)
    print(class_balance)

    print("\n--- 3. UNIQUE BUILDING COUNT PER CLASS PER SPLIT ---")
    building_balance = df.groupby(['split', 'style_label'])['building_id'].nunique().unstack(fill_value=0)
    print(building_balance)

    print("\n--- 4. FILE READABILITY CHECK ---")
    missing_files = []
    project_root = os.path.dirname(os.path.dirname(processed_manifest_path))
    for idx, row in df.iterrows():
        full_path = os.path.join(project_root, row['processed_path'])
        if not os.path.exists(full_path):
            missing_files.append(full_path)

    if missing_files:
        print(f"  [FAIL] Found {len(missing_files)} missing processed files! First 5: {missing_files[:5]}")
    else:
        print(f"  [PASS] All {len(df)} processed images exist on disk.")

    print("\n==================================================")
    print("Sanity checks finished.")

if __name__ == "__main__":
    run_sanity_checks()
