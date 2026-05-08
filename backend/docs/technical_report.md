# BÁO CÁO KỸ THUẬT: PHÂN TÍCH CẢM XÚC ĐA MIỀN VÀ ĐA NGÔN NGỮ

Báo cáo này tổng hợp toàn bộ kiến trúc, dữ liệu và kết quả thực nghiệm thực tế từ hệ thống.

---

## 1. Giới thiệu
Dự án giải quyết bài toán phân tích cảm xúc thông qua 3 chặng thực nghiệm chính (12 kịch bản), tập trung vào khả năng thích ứng của mô hình khi có sự thay đổi về miền (Domain) và ngôn ngữ (Language) trong điều kiện ít dữ liệu.

---

## 2. Công cụ và Môi trường phát triển
- **Framework**: PyTorch & HuggingFace Transformers.
- **Hardware**: Huấn luyện trên GPU NVIDIA với CUDA để đạt tốc độ xử lý ~1.43 it/s trong mạng DANN.
- **Monitoring**: Sử dụng `tqdm` để theo dõi tiến trình và `EarlyStopping` để tối ưu thời gian huấn luyện.

---

## 3. Các kỹ thuật NLP và Huấn luyện tiên tiến áp dụng
Dựa trên mã nguồn và kết quả chạy thực tế, các kỹ thuật sau đã được triển khai thành công:
1.  **DANN & GRL**: Sử dụng mạng đối nghịch để đạt **Macro F1-score 0.8922** trên kịch bản Multidomain (S3).
2.  **Early Stopping**: Cơ chế tự động dừng huấn luyện sau 3 epoch không cải thiện để tránh Overfitting (đã kích hoạt thành công ở Epoch 4 của S3).
3.  **Multi-task Learning (MTL)**: Tích hợp đầu phân loại Miền và Ngôn ngữ cùng với tác vụ Cảm xúc chính.
4.  **Zero-shot Cross-lingual Transfer**: Khả năng dự đoán cảm xúc trên tiếng Việt dù chỉ được huấn luyện trên tiếng Anh và tiếng Pháp.
5.  **Few-shot Fine-tuning (S4b)**: Tinh chỉnh mô hình đa ngôn ngữ với một lượng dữ liệu cực nhỏ (500 mẫu) để tối ưu hóa cho ngôn ngữ mục tiêu.
6.  **Unsupervised Domain Adaptation**: Sử dụng dữ liệu miền đích (Target) không có nhãn để căn chỉnh (Alignment) không gian vector.
7.  **Adversarial Alignment Visualization**: Sử dụng **t-SNE** để trực quan hóa sự hội tụ của dữ liệu Source và Target.

---

## 4. Kiến trúc mô hình theo chặng (Pipeline)

### Chặng 1: Multidomain (S3 - DANN)
- **Source**: IMDb + Yelp (Labeled).
- **Target**: Amazon English (Unlabeled).
- **Kết quả thực tế**: Accuracy: **0.89**, Macro F1: **0.89**. Mô hình thể hiện sự cân bằng tuyệt vời giữa Precision và Recall trên cả hai lớp cảm xúc.

### Chặng 2: Multilingual (S4, S5, S6)
- **Source**: English + French/Vietnamese.
- **Target**: Vietnamese Student Feedback (VSFC).
- **Kỹ thuật**: Kết hợp giữa Zero-shot và Google Translation Baseline.

### Chặng 3: Unified Framework (S8, S9)
- **Mô hình**: `AdvancedMultiTaskModel` với 3 đầu ra song song, giải quyết bài toán "Double Shift" (lệch cả miền và ngôn ngữ).

---

## 5. Phân tích kết quả thực nghiệm (S3)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **0 (Negative)** | 0.87 | 0.93 | 0.90 | 1019 |
| **1 (Positive)** | 0.92 | 0.86 | 0.89 | 981 |
| **Macro Avg** | **0.89** | **0.89** | **0.89** | 2000 |

**Nhận xét**: Kết quả F1-score 0.90 cho lớp Negative và 0.89 cho lớp Positive cho thấy mạng DANN đã học được các đặc trưng cực kỳ ổn định, không bị thiên lệch dù miền dữ liệu thay đổi từ Phim ảnh sang Thương mại điện tử.

---

## 7. Kết luận
Dự án đã đạt được các mục tiêu đề ra với kết quả thực nghiệm rất khả quan. Việc áp dụng các kỹ thuật đối nghịch (Adversarial) và dừng sớm (Early Stopping) đã giúp mô hình đạt được độ chính xác cao và khả năng tổng quát hóa tốt trên nhiều miền dữ liệu và ngôn ngữ khác nhau.
