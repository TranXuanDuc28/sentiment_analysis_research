# Robust Sentiment Analysis: Multidomain & Multilingual Framework

Dự án nghiên cứu về phân tích cảm xúc (Sentiment Analysis) trong điều kiện có sự dịch chuyển về miền dữ liệu (Domain Shift) và ngôn ngữ (Language Shift).

## 🚀 Hướng dẫn chạy thực nghiệm

### 1. Cài đặt môi trường
Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Cấu hình thực nghiệm
Tất cả các tham số được cấu hình trong tệp `config.yaml`. Bạn có thể điều chỉnh:
- `epochs`, `batch_size`, `learning_rate`
- Các kịch bản muốn chạy (đặt `true/false`)

### 3. Cách chạy kịch bản
Bạn có thể chạy toàn bộ kịch bản hoặc từng kịch bản riêng lẻ bằng `main.py`:

- **Chạy toàn bộ (S1 -> S7):**
  ```bash
  python main.py --s 0
  ```

- **Chạy từng Phase cụ thể:**
  - **Phase 1 (Multidomain):** `python main.py --s 1` (hoặc 2, 3)
  - **Phase 2 (Multilingual):** `python main.py --s 4` (hoặc 5, 6)
  - **Phase 3 (Combined):** `python main.py --s 7`

### 4. Xem kết quả
- **Kết quả JSON:** Lưu tại thư mục `results/`.
- **Biểu đồ so sánh:** Sau khi chạy, hệ thống tự động tạo các biểu đồ tại `results/plots/`.
- **Báo cáo tóm tắt:** Xem file `results/research_summary.md` để thấy bảng tổng hợp kết quả F1-Macro.

## 📊 Cấu trúc thực nghiệm (Narrative)

Hệ thống được thiết kế theo 3 giai đoạn nghiên cứu tăng dần về độ khó:

1.  **Phase 1 (RQ1 - Domain Robustness):** Tập trung vào khả năng chống chịu của mô hình trước sự thay đổi miền dữ liệu (IMDb -> Amazon). S1 được thiết lập làm mốc **Zero-shot Domain Transfer**.
2.  **Phase 2 (RQ2 - Multilingual Robustness):** Mở rộng từ sự ổn định về miền sang sự ổn định về ngôn ngữ (EN -> VI). Các kịch bản được thực hiện để đo lường khả năng chuyển đổi tri thức cross-lingual.
3.  **Phase 3 (RQ3 - Unified Robustness Framework):** Kết hợp đồng thời mọi yếu tố dịch chuyển vào một khung mô hình hợp nhất duy nhất để đạt được độ "Robust" tối đa.

## 🛠️ Yêu cầu hệ thống
- Python 3.8+
- GPU (Khuyến khích) để huấn luyện các mô hình XLM-RoBERTa.
