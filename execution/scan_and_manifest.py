import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

raw_dir = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\raw_data"
output_csv = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\manifest.csv"

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

def parse_folder_structure(file_path):
    rel_path = os.path.relpath(file_path, raw_dir)
    parts = rel_path.split(os.sep)
    
    if len(parts) < 2:
        return None, None
        
    first_dir = parts[0]
    
    if first_dir == "HistoricVietnam-OldPics":
        if len(parts) >= 3:
            building_id = f"OldPics_{parts[1]}"
            style_label = parts[1].split('.')[0] if '.' in parts[1] else parts[1]
            return building_id, style_label
        else:
            return "HistoricVietnam-OldPics-Misc", "A1"
            
    elif first_dir == "Other-Modernism":
        if len(parts) >= 3:
            building_id = f"OtherModernism_{parts[1]}"
            return building_id, "B2"
        else:
            return "Other-Modernism", "B2"
            
    else:
        building_id = first_dir
        style_label = building_id.split('.')[0] if '.' in building_id else building_id
        return building_id, style_label

def scan_files():
    # Make sure .tmp folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    data = []
    print("Scanning raw files and verifying images...")
    
    all_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                all_files.append(os.path.join(root, f))
                
    for path in tqdm(all_files):
        building_id, style_label = parse_folder_structure(path)
        if not building_id or not style_label:
            continue
            
        try:
            with Image.open(path) as img:
                img.verify()
            is_valid = True
        except Exception as e:
            print(f"Corrupt image skipped: {path} - {e}")
            is_valid = False
            
        if is_valid:
            rel_path = os.path.relpath(path, start=os.path.dirname(raw_dir))
            rel_path = rel_path.replace(os.sep, '/')
            data.append({
                'file_path': rel_path,
                'filename': os.path.basename(path),
                'building_id': building_id,
                'style_label': style_label
            })
            
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Created manifest with {len(df)} images at: {output_csv}")
    print("\nClass distribution in raw manifest:")
    print(df['style_label'].value_counts())
    print("\nUnique buildings count per class:")
    print(df.groupby('style_label')['building_id'].nunique())

if __name__ == "__main__":
    scan_files()
