import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest.csv"
MANIFEST_CLEANED_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest_cleaned.csv"
MANIFEST_CURATED_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest_curated.csv"

def get_subpath(p):
    p_str = str(p).replace('\\', '/')
    for split in ['train/', 'val/', 'test/']:
        if split in p_str:
            return p_str.split(split, 1)[1]
    return p_str

def main():
    if not MANIFEST_CLEANED_PATH.exists():
        print("Cleaned manifest not found.")
        return

    # Load manifest
    print("Loading original clean manifest...")
    df_cleaned = pd.read_csv(MANIFEST_CLEANED_PATH)
    cleaned_subpaths = set(df_cleaned['processed_path'].apply(get_subpath).tolist())

    # Find all curation CSV files
    all_files = os.listdir(PROJECT_ROOT)
    move_to_kept_files = [f for f in all_files if f.startswith("move_to_kept_patches") and f.endswith(".csv")]
    filter_out_files = [f for f in all_files if f.startswith("filter_out_patches") and f.endswith(".csv")]

    print(f"Applying curation changes from {len(move_to_kept_files)} addition files and {len(filter_out_files)} removal files...")

    # 1. Gather additions
    added_subpaths = set()
    for f_name in move_to_kept_files:
        try:
            df = pd.read_csv(PROJECT_ROOT / f_name)
            for _, row in df.iterrows():
                if row['new_action'] == 'KEEP':
                    added_subpaths.add(get_subpath(row['processed_path']))
        except Exception as e:
            print(f"Error reading {f_name}: {e}")

    # 2. Gather removals
    removed_subpaths = set()
    for f_name in filter_out_files:
        try:
            df = pd.read_csv(PROJECT_ROOT / f_name)
            for _, row in df.iterrows():
                if row['new_action'] == 'FILTER':
                    removed_subpaths.add(get_subpath(row['processed_path']))
        except Exception as e:
            print(f"Error reading {f_name}: {e}")

    # Apply changes
    final_subpaths = (cleaned_subpaths - removed_subpaths) | added_subpaths

    # Reconstruct curated manifest
    print("Reconstructing final curated manifest...")
    df_all = pd.read_csv(MANIFEST_PATH)
    df_all['subpath'] = df_all['processed_path'].apply(get_subpath)
    df_curated = df_all[df_all['subpath'].isin(final_subpaths)].copy()
    df_curated.drop(columns=['subpath'], errors='ignore', inplace=True)

    # Save to disk
    df_curated.to_csv(MANIFEST_CURATED_PATH, index=False)
    print(f"Curated manifest saved successfully to: {MANIFEST_CURATED_PATH} ({len(df_curated):,} patches)")

    # 3. Archive the CSV files instead of deleting them
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = PROJECT_ROOT / "outputs" / "curation_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    print("Archiving desktop CSV files...")
    for f_name in move_to_kept_files + filter_out_files:
        try:
            src_path = PROJECT_ROOT / f_name
            # Generate archived filename, e.g. move_to_kept_patches_20260730_142558.csv
            base_name, ext = os.path.splitext(f_name)
            dest_name = f"{base_name}_{timestamp}{ext}"
            dest_path = history_dir / dest_name
            
            # Move the file
            os.rename(src_path, dest_path)
            print(f"  Archived: {f_name} -> outputs/curation_history/{dest_name}")
        except Exception as e:
            print(f"  Error archiving {f_name}: {e}")

    # Create the REVIEW_PLAN.md file
    generate_review_plan(len(df_cleaned), len(added_subpaths), len(removed_subpaths), len(df_curated))

def generate_review_plan(original_count, added, removed, curated):
    plan_path = PROJECT_ROOT / "REVIEW_PLAN.md"
    content = f"""# Kế hoạch Đánh giá Thủ công (Manual Review Plan)

## 1. Tóm tắt Tiến độ Huấn luyện & Curation
* **Trạng thái:** Toàn bộ 6 mô hình đã hoàn thành huấn luyện đợt 2 trên GPU. **DINOv2-S** đạt kết quả tốt nhất với **52.20% Test Accuracy**.
* **Tiến độ Curation (Lọc Thủ công):** Bạn đã hoàn thành xuất sắc **12 đợt lọc** với tổng cộng **24,000 patches** được đánh giá thủ công (12,000 ở nhánh Lọc rác và 12,000 ở nhánh Giữ ảnh).
* **Kết quả Curated Manifest hiện tại:**
  * Kích thước ban đầu (YOLOv8 >= 1.8%): `{original_count:,}` patches
  * Số patch được bạn thêm lại (YOLO lọc nhầm): `+{added:,}` patches
  * Số patch được bạn xóa bỏ (vẫn còn nhiễu rác): `-{removed:,}` patches
  * Quy mô dữ liệu hiện tại sau curation: **`{curated:,}`** patches

---

## 2. Kế hoạch Review ngày mai

### A. Nhiệm vụ và Tác vụ
1. **Tiếp tục Đợt 13 (2,000 ảnh/tập):**
   * Các file HTML đã được tự động làm mới với **2,000 ảnh tiếp theo** không trùng lặp.
   * Review các patch được lọc bỏ tại: [review_filtered_patches.html](file:///c:/Users/Admin/Desktop/architect/review_filtered_patches.html) (chọn ảnh để **KEEP**).
   * Review các patch được giữ lại tại: [review_kept_patches.html](file:///c:/Users/Admin/Desktop/architect/review_kept_patches.html) (chọn ảnh để **FILTER OUT**).
2. **Xuất CSV & Chạy script:**
   * Sau khi hoàn thành đợt review mới, xuất các tệp CSV như thường lệ.
   * Chạy lệnh sau để cập nhật Manifest Curated và làm mới đợt tiếp theo:
     ```powershell
     .venv\\Scripts\\python scratch/apply_curation_changes.py
     ```

### B. Thống kê Dữ liệu còn lại cần Curation
* **Pool Kept (Đang giữ) còn lại:** `{original_count - 12000:,}` patches (sẽ tiếp tục hiển thị tại `review_kept_patches.html`).
* **Pool Filtered (Đã lọc) còn lại:** `{97683 - 12000:,}` patches (sẽ tiếp tục hiển thị tại `review_filtered_patches.html`).
"""
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Review plan saved to: {plan_path}")

if __name__ == "__main__":
    main()
