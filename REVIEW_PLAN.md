# Kế hoạch Đánh giá Thủ công (Manual Review Plan)

## 1. Tóm tắt Tiến độ Huấn luyện & Curation
* **Trạng thái:** Toàn bộ 6 mô hình đã hoàn thành huấn luyện đợt 2 trên GPU. **DINOv2-S** đạt kết quả tốt nhất với **52.20% Test Accuracy**.
* **Tiến độ Curation (Lọc Thủ công):** Bạn đã hoàn thành xuất sắc **12 đợt lọc** với tổng cộng **24,000 patches** được đánh giá thủ công (12,000 ở nhánh Lọc rác và 12,000 ở nhánh Giữ ảnh).
* **Kết quả Curated Manifest hiện tại:**
  * Kích thước ban đầu (YOLOv8 >= 1.8%): `85,991` patches
  * Số patch được bạn thêm lại (YOLO lọc nhầm): `+1,444` patches
  * Số patch được bạn xóa bỏ (vẫn còn nhiễu rác): `-414` patches
  * Quy mô dữ liệu hiện tại sau curation: **`87,021`** patches

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
     .venv\Scripts\python scratch/apply_curation_changes.py
     ```

### B. Thống kê Dữ liệu còn lại cần Curation
* **Pool Kept (Đang giữ) còn lại:** `73,991` patches (sẽ tiếp tục hiển thị tại `review_kept_patches.html`).
* **Pool Filtered (Đã lọc) còn lại:** `85,683` patches (sẽ tiếp tục hiển thị tại `review_filtered_patches.html`).
