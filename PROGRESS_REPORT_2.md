# Báo cáo tiến độ - Đợt 2 (Huấn luyện với Dữ liệu Lọc Sạch YOLOv8 - 86,000 Patches)

## 1. Bối cảnh và Kế hoạch Đợt 2

Trong đợt huấn luyện thứ 1, mặc dù chúng ta đã chạy thành công trên GPU và cải thiện tốc độ tải dữ liệu lên **4.4x**, hiệu năng thực tế của các mô hình trên tập Test chỉ đạt quanh mức **41% - 45%**. Qua phân tích chuyên sâu, chúng tôi nhận thấy nguyên nhân cốt lõi là **nhiễu dữ liệu cực kỳ lớn**:
* Hàng ngàn patch lưới $1000 \times 1000$ bị cắt mù, chỉ chứa khoảng trời trống, mảng tường gạch trơn, giàn giáo xây dựng hoặc mặt đường nhựa, nhưng vẫn bị gán nhãn phong cách của công trình gốc.
* Mô hình bị ép phải học các pixel nhiễu không mang thông tin kiến trúc, dẫn đến overfitting trên tập huấn luyện (accuracy > 98%) nhưng suy giảm nặng ở tập test.

### Điều chỉnh Kế hoạch và Ngưỡng Lọc (Relaxed Sifting):
* Ở lần chạy thử đầu tiên của Đợt 2, khi áp đặt ngưỡng lọc diện tích kiến trúc của YOLOv8 rất nghiêm ngặt (>= 70%), tập dữ liệu bị thu hẹp quá mức xuống chỉ còn **2,402 patches** (gây thiếu hụt trầm trọng dữ liệu, khiến mô hình dễ overfit).
* Để giải quyết vấn đề này, chúng ta đã hạ ngưỡng lọc xuống **>= 1.8%**. Ngưỡng này vừa đủ để **loại bỏ hoàn toàn hơn 97,000 patches rác** (những patch chứa 0% hoặc gần như 0% thông tin tòa nhà như bầu trời trống, tán cây rậm, mặt đường nhựa), vừa giữ lại **85,991 patches** có chứa các thành phần kiến trúc để huấn luyện mô hình mạnh mẽ hơn.

---

## 2. Kết quả Sàng lọc Dữ liệu bằng YOLOv8

Chúng tôi đã chạy lọc trên toàn bộ **183,674 patch**:
* **Tỷ lệ lọc:** Giữ lại **85,991 patches** chất lượng (đạt tỷ lệ giữ lại **46.81%**).
* **Phân bố patch sạch theo lớp kiến trúc:**
  * **A1 (French Colonial):** 36,715 patch
  * **B1 (Vernacular):** 27,755 patch
  * **A2 (Modernism):** 11,514 patch
  * **B2 (Eclectic):** 10,007 patch
  * **Tổng cộng:** 85,991 patch (chia stratified theo tỷ lệ 70-15-15 ở cấp độ `building_id` thành Train: 56,050, Val: 16,382, Test: 13,559).

---

## 3. Kết quả Thực nghiệm Huấn luyện Đợt 2 (86,000 Patches)

Toàn bộ 6 mô hình đã được huấn luyện hoàn chỉnh trên GPU với tập dữ liệu lọc sạch (sử dụng Cosine Annealing, AdamW, độ chính xác hỗn hợp FP16 và cơ chế Early Stopping với patience = 3). Dưới đây là bảng so sánh hiệu năng chi tiết trên tập test sạch (13,559 patches):

### 3.1. Bảng so sánh tổng hợp các mô hình (86k Cleaned Dataset)

| Mô hình | Loại kiến trúc | Số tham số | Độ phức tạp | Train Acc *(Tốt nhất)* | Test Accuracy *(Chính)* | Test Macro-F1 *(Chính)* | Tốc độ suy luận |
|---|---|---:|---:|---:|---:|---:|---:|
| **DINOv2-S** 🏆 | Transformer (Self-Supervised) | 22.0M | 4.6G | 67.62% | **52.20%** | **0.4980** | **4.2 ms/ảnh** |
| **ResNet-50** | CNN (Baseline) | 25.6M | 4.1G | 84.42% | **51.72%** | **0.4950** | **4.3 ms/ảnh** |
| **EfficientNet-V2-S** | CNN (Fused-MBConv) | 21.5M | 2.9G | 93.76% | **48.14%** | **0.4526** | **4.3 ms/ảnh** |
| **ConvNeXt-Tiny** | Modern CNN | 28.6M | 4.5G | 99.49% | **44.42%** | **0.3964** | **4.3 ms/ảnh** |
| **ViT-B/16** | Transformer (Flat Patch) | 86.0M | 17.6G | 95.42% | **43.54%** | **0.4042** | **7.4 ms/ảnh** |
| **Swin-V2-T** | Transformer (Hierarchical Window) | 28.0M | 4.5G | 94.01% | **42.89%** | **0.3877** | **6.7 ms/ảnh** |

### 3.2. Bảng chỉ số F1-score theo từng lớp kiến trúc

| Mô hình | Lớp A1 (French) | Lớp A2 (Modernism) | Lớp B1 (Vernacular) | Lớp B2 (Eclectic) |
|---|---:|---:|---:|---:|
| **DINOv2-S** 🏆 | **0.6922** | 0.4169 | **0.5881** | 0.2947 |
| **ResNet-50** | 0.6143 | **0.4428** | 0.5728 | **0.3499** |
| **EfficientNet-V2-S** | 0.5738 | 0.4059 | 0.5344 | 0.2962 |
| **ViT-B/16** | 0.5240 | 0.4342 | 0.4827 | 0.1761 |
| **ConvNeXt-Tiny** | 0.5392 | 0.3794 | 0.5065 | 0.1603 |
| **Swin-V2-T** | 0.5411 | 0.3725 | 0.4658 | 0.1715 |

---

## 4. Phân tích Nhận xét Chuyên sâu & Hạn chế

1. **Sự hồi sinh ngoạn mục của DINOv2-S:**
   * Trong lần chạy dữ liệu cực sạch (chỉ 2.4k patches), DINOv2-S bị sụp đổ hoàn toàn (Test Acc đạt 10.00%, F1 đạt 0.04). Tuy nhiên, khi tăng quy mô tập dữ liệu lên **86,000 patches**, DINOv2-S đã phát huy sức mạnh tối đa của bộ trích xuất đặc trưng tự giám sát (Self-Supervised), vươn lên dẫn đầu toàn bảng với **52.20% Test Accuracy** và **0.4980 Macro-F1**.
   * DINOv2-S cho thấy khả năng tổng quát hóa xuất sắc nhất khi chỉ cần Train Acc ở mức **67.62%** đã đạt Test Acc **52.20%** (khoảng cách overfitting rất hẹp).

2. **Overfitting nghiêm trọng ở các mô hình dung lượng lớn:**
   * **ConvNeXt-Tiny** đạt Train Acc cực kỳ cao (**99.49%**) nhưng Test Acc chỉ đạt **44.42%**.
   * Tương tự, **ViT-B/16** (Train Acc: 95.42% vs Test Acc: 43.54%) và **Swin-V2-T** (Train Acc: 94.01% vs Test Acc: 42.89%) đều bị overfitting rất nặng. Điều này cho thấy mặc dù đã lọc bớt 97k patch nhiễu trống, dữ liệu ảnh kiến trúc thực tế vẫn còn chứa các yếu tố nền (background context) trùng lặp dễ khiến các mô hình transformer dung lượng lớn học thuộc lòng thay vì học đặc trưng hình học.

3. **ResNet-50 duy trì phong độ CNN baseline cực tốt:**
   * ResNet-50 đạt vị trí thứ hai toàn bảng với **51.72% Test Accuracy**, chứng tỏ kiến trúc CNN truyền thống với cơ chế bias cảm nhận (inductive bias) cục bộ vẫn hoạt động cực kỳ ổn định trên tập dữ liệu dạng patch của công trình kiến trúc.

---

## 5. Kế hoạch Huấn luyện Đợt 3 hướng tới mục tiêu > 80% Accuracy

Dựa trên các kết quả và phân tích ở Đợt 2, chúng tôi đề xuất các giải pháp chiến lược cho Đợt 3 để hướng tới mục tiêu **> 80% Accuracy**:

1. **Biểu quyết cấp độ Tòa nhà (Building-Level Majority Voting):**
   * Hiện tại, tất cả các mô hình đều được đánh giá ở cấp độ Patch-Level (tập test gồm 13,559 patches riêng lẻ). Khi gom các patches lại theo từng công trình (`building_id`) và thực hiện biểu quyết số đông (Majority Voting), độ chính xác thực tế trên từng tòa nhà dự kiến sẽ tăng mạnh (vượt mốc 75%-80%) vì biểu quyết sẽ tự động triệt tiêu các dự đoán sai biệt lập của các patch nhiễu.

2. **Tăng cường Regularization chống Overfitting:**
   * Áp dụng mạnh mẽ **Weight Decay (0.05)**, **DropPath / Stochastic Depth (0.2)** đối với ViT và Swin-V2.
   * Tích hợp các kỹ thuật tăng cường dữ liệu nâng cao như **Mixup** và **CutMix** trực tiếp vào pipeline huấn luyện để ngăn các mô hình CNN/ViT dung lượng lớn học thuộc lòng.

3. **Huấn luyện với độ phân giải lớn hơn:**
   * Nén ảnh $1000 \times 1000$ về $224 \times 224$ làm mất đi nhiều chi tiết chạm khắc phào chỉ, cửa sổ. Trong đợt tới, chúng ta sẽ nâng độ phân giải huấn luyện lên **`384x384`** cho các mô hình tốt nhất (DINOv2-S và ResNet-50).

4. **Tiêu chí Lọc Dữ liệu Đợt Cuối (Manual Curation Criteria):**
   * **Tiêu chí 1 (Thông tin kiến trúc):** Phần lớn diện tích của patch phải chứa thông tin kiến trúc rõ ràng (tường, ngói, hoa văn, màu sơn, chất liệu phải hiển thị rõ ràng và có thể mang một phong cách nhất định).
   * **Tiêu chí 2 (Hạn chế nhiễu - Noise Control):** Không chứa quá nhiều nhiễu làm ảnh hưởng đến nhận diện (nhiễu bao gồm: một phần kiến trúc của tòa nhà/công trình khác, thùng rác, xe cộ, người, cây lá không thuộc công trình kiến trúc hoặc không giúp định nghĩa phong cách/style).
   * **Tiêu chí 3 (Chất lượng ảnh):** Loại bỏ các ảnh gặp lỗi chất lượng như quá sáng, quá tối, hoặc quá mờ.

