import os
import shutil
from pathlib import Path

def setup_kaggle_dataset():
    # 1. Define paths
    input_base = Path("/kaggle/input")
    working_base = Path("/kaggle/working")
    target_dir = working_base / "processed_data"

    # Find the dataset folder inside /kaggle/input
    dataset_dirs = [d for d in input_base.iterdir() if d.is_dir() and ("archi" in d.name.lower())]
    if not dataset_dirs:
        print("Error: Could not find dataset folder matching 'archi' in /kaggle/input")
        print(f"Available input folders: {[d.name for d in input_base.iterdir()]}")
        return
    
    dataset_dir = dataset_dirs[0]
    print(f"Found dataset directory: {dataset_dir}")

    # Remove target directory if it already exists to avoid conflict on retry
    if target_dir.exists():
        print(f"Removing existing {target_dir}...")
        shutil.rmtree(target_dir)

    # 2. Define the mapping of Kaggle Input subdirs -> Target standard splits
    # Format: (Source Path relative to dataset_dir, Target Path relative to processed_data)
    mappings = [
        # Test split
        ("test/test/A1", "test/A1"),
        ("test/test/A2", "test/A2"),
        ("test/test/B1", "test/B1"),
        ("test/test/B2", "test/B2"),
        
        # Val split
        ("val/val/A1", "val/A1"),
        ("val/val/A2", "val/A2"),
        ("val/val/B1", "val/B1"),
        ("val/val/B2", "val/B2"),
        
        # Train split (from split zip files)
        ("train_A1/train/A1", "train/A1"),
        ("train_A2/train/A2", "train/A2"),
        ("train_B1/train/B1", "train/B1"),
        ("train_B2/train/B2", "train/B2"),
    ]

    print("\nCreating symbolic links...")
    for src_rel, dest_rel in mappings:
        src_path = dataset_dir / src_rel
        dest_path = target_dir / dest_rel

        if not src_path.exists():
            print(f"Warning: Source path does not exist: {src_path}")
            continue

        # Create parent directories for target link
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Create symlink
        try:
            os.symlink(src_path, dest_path)
            print(f"Linked: {src_rel} -> {dest_rel}")
        except Exception as e:
            print(f"Error linking {src_rel} to {dest_rel}: {e}")

    # 3. Verify the setup
    print("\n=== Verification ===")
    total_files = 0
    for split in ["train", "val", "test"]:
        split_dir = target_dir / split
        if split_dir.exists():
            count = sum(1 for root, dirs, files in os.walk(split_dir) for f in files)
            print(f"Split '{split}': {count} files verified.")
            total_files += count
        else:
            print(f"Warning: Split '{split}' is missing or incomplete.")
    print(f"Total files in linked dataset: {total_files}")

if __name__ == "__main__":
    setup_kaggle_dataset()
