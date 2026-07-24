"""
prepare_split_uploads.py — Split dataset into smaller zip files for easier cloud upload.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed_data"

def zip_folder_to_archive(source_dir: Path, zip_file_path: Path, relative_to_dir: Path):
    print(f"Compressing {source_dir.relative_to(relative_to_dir)} into {zip_file_path.name}...")
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                archive_name = file_path.relative_to(relative_to_dir)
                zipf.write(file_path, archive_name)
    print(f"Finished: {zip_file_path.name} ({zip_file_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    if not PROCESSED_DIR.exists():
        print(f"Error: {PROCESSED_DIR} does not exist.")
        exit(1)

    # 1. Compress test
    zip_folder_to_archive(PROCESSED_DIR / "test", PROJECT_ROOT / "test.zip", PROCESSED_DIR)

    # 2. Compress val
    zip_folder_to_archive(PROCESSED_DIR / "val", PROJECT_ROOT / "val.zip", PROCESSED_DIR)

    # 3. Compress train splits class by class
    for cls in ["A1", "A2", "B1", "B2"]:
        class_dir = PROCESSED_DIR / "train" / cls
        if class_dir.exists():
            zip_file = PROJECT_ROOT / f"train_{cls}.zip"
            zip_folder_to_archive(class_dir, zip_file, PROCESSED_DIR)

    print("\n=== SPLIT PREPARATION COMPLETE ===")
    print("Upload the following 6 zip files into the same Kaggle Dataset:")
    print("1. test.zip")
    print("2. val.zip")
    print("3. train_A1.zip")
    print("4. train_A2.zip")
    print("5. train_B1.zip")
    print("6. train_B2.zip")
