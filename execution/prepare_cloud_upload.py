"""
prepare_cloud_upload.py — Helper script to package processed patches and training scripts for Google Colab/Kaggle.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
ZIP_OUT_PATH = PROJECT_ROOT / "processed_data.zip"
SCRIPTS_ZIP_PATH = PROJECT_ROOT / "training_scripts.zip"

def zip_directory(folder_path: Path, zip_path: Path):
    print(f"Compressing {folder_path.name} into {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                # Save relative to the parent of processed_data to maintain folder structure
                archive_name = file_path.relative_to(folder_path.parent)
                zipf.write(file_path, archive_name)
    print(f"Successfully created {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

def package_scripts():
    print("Packaging training scripts...")
    training_folder = PROJECT_ROOT / "execution" / "training"
    with zipfile.ZipFile(SCRIPTS_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in ["train.py", "dataset.py", "evaluate.py", "compare_models.py"]:
            file_path = training_folder / f
            if file_path.exists():
                zipf.write(file_path, f)
    print(f"Successfully packaged scripts into {SCRIPTS_ZIP_PATH.name}")

if __name__ == "__main__":
    if not PROCESSED_DIR.exists():
        print(f"Error: {PROCESSED_DIR} does not exist. Run preprocessing first.")
    else:
        zip_directory(PROCESSED_DIR, ZIP_OUT_PATH)
        package_scripts()
        print("\n=== PREPARATION COMPLETE ===")
        print("Upload the following files to Google Drive or Kaggle:")
        print(f"1. Data: {ZIP_OUT_PATH}")
        print(f"2. Scripts: {SCRIPTS_ZIP_PATH}")
