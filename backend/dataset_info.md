# Danh mục Dữ liệu: Cross-Domain & Cross-Lingual Sentiment Analysis

Tài liệu này mô tả chi tiết các tập dữ liệu được sử dụng trong hệ thống phân tích cảm xúc Đa miền (Multidomain) và Đa ngôn ngữ (Multilingual). Toàn bộ hệ thống được thiết kế để giả lập môi trường **Low-Resource** (Giới hạn 1000 mẫu cho mỗi miền) nhằm làm nổi bật sức mạnh của Transfer Learning và Domain Adaptation.

---

## 1. Dữ liệu Tiếng Anh (English Datasets) - Phục vụ Multidomain
Nhóm dữ liệu này dùng để thiết lập baseline, huấn luyện mô hình cơ sở và đánh giá hiện tượng Domain Shift (lệch miền).

### 1.1. IMDb (Internet Movie Database)
*   **Miền (Domain):** Phim ảnh (Movie Reviews).
*   **Vai trò:** Là miền Nguồn (Source Domain) chính. Dữ liệu văn bản dài, ngôn từ phong phú, cấu trúc câu phức tạp.
*   **Ánh xạ nhãn:** `0` (Negative), `1` (Positive).

### 1.2. Yelp Reviews
*   **Miền (Domain):** Nhà hàng / Dịch vụ (Restaurant Reviews).
*   **Vai trò:** Làm miền Nguồn bổ sung (Multi-source) khi kết hợp cùng IMDb để tạo ra một mô hình có khả năng tổng quát hóa cao.
*   **Ánh xạ nhãn:** `0` (Negative), `1` (Positive).

### 1.3. Amazon Reviews Multi (MARC)
*   **Miền (Domain):** Thương mại điện tử (Product Reviews - Books, Electronics...).
*   **Ngôn ngữ:** Mặc dù MARC hỗ trợ 6 ngôn ngữ (EN, JA, DE, FR, ES, ZH), trong dự án này chúng ta chỉ sử dụng phần **Tiếng Anh (EN)**.
*   **Vai trò:** Đóng vai trò là miền Đích (Target Domain) để đánh giá khả năng Thích ứng miền (Domain Adaptation) bằng mạng DANN.
*   **Ánh xạ nhãn:**
    *   `1 - 2 stars`: 0 (Negative)
    *   `3 stars`: Bị loại bỏ (Neutral)
    *   `4 - 5 stars`: 1 (Positive)

---

## 2. Dữ liệu Tiếng Việt (Vietnamese Datasets) - Phục vụ Multilingual
Nhóm dữ liệu này dùng để đánh giá khả năng chuyển giao tri thức chéo ngôn ngữ (Cross-lingual Transfer) từ Tiếng Anh sang Tiếng Việt.

### 2.1. UIT-VSFC (Vietnamese Students' Feedback Corpus)
*   **Nguồn:** Đại học CNTT - ĐHQG TP.HCM (UIT).
*   **Miền (Domain):** Giáo dục / Phản hồi của sinh viên.
*   **Vai trò:** Đóng vai trò là miền Đích (Target Domain) cho cả hai bài toán: Đa ngôn ngữ (chỉ lệch ngôn ngữ) và Tổng hợp (lệch cả ngôn ngữ lẫn miền).
*   **Ánh xạ nhãn:**
    *   `0`: Negative
    *   `1`: Neutral (Bị loại bỏ để quy về Binary)
    *   `2`: Positive -> Map thành `1`

---

## 3. Cấu hình Kích thước Dữ liệu (Low-Resource Setup)

Hệ thống giới hạn nghiêm ngặt số lượng mẫu thông qua tham số cấu hình trong `config.yaml` (`max_samples_train` và `max_samples_test` mặc định = 1000).

*   **Train / Validation:** Khi cấu hình 1000 mẫu, hệ thống tự động tách theo tỷ lệ 9:1 (900 mẫu Train, 100 mẫu Validation).
*   **Test:** Cố định 1000 mẫu để đảm bảo tính công bằng khi so sánh giữa các miền và ngôn ngữ.
*   **Multi-source:** Khi trộn nhiều nguồn (ví dụ IMDb + Yelp), hệ thống tự động chia đều (500 mẫu IMDb + 500 mẫu Yelp) để tổng số lượng không vượt quá giới hạn.
*   **Unlabeled Target (Cho DANN):** Sử dụng 1000 mẫu nhưng loại bỏ hoàn toàn nhãn cảm xúc (gán = -1), chỉ giữ lại nhãn Miền/Ngôn ngữ.

---

## 4. Phân bổ Dữ liệu trong các Kịch bản Thực nghiệm (14 Scenarios)

| # | Nhóm| Kịch bản | feature area | Dữ liệu Nguồn | Dữ liệu Đích | Thách thức chính |
|---|---|---|---|---|
| **S1** | **Baseline Multidomain** | IMDb, Yelp | Amazon (EN) | Domain Shift (Phim -> Thương mại) |
| **S2** | **Multi-task MD** | IMDb, Yelp | Amazon (EN) | Học đa nhiệm để tổng quát hóa |
| **S3** | **DANN MD** | IMDb, Yelp | Amazon (EN) | Căn chỉnh không gian vector miền |
| **S4** | **Zero-shot ML** | IMDb (EN), Amazon (FR) | VSFC (VI) | Language Shift (EN/FR -> VI) |
| **S4b** | **Few-shot ML** | model S4 + 500 VSFC | VSFC (VI) | Cải tiến S4 bằng Fine-tuning |
| **S5** | **Translation ML** | IMDb (EN) | VSFC -> EN | Dịch máy (Machine Translation) |
| **S6** | **Joint Training** | EN, FR, VI | VSFC (VI) | Huấn luyện đồng thời đa ngôn ngữ |
| **S7** | **Unified Zero-shot**| IMDb (EN) | VSFC (VI) | Double Shift (Domain + Language) |
| **S8** | **Unified DANN** | EN + FR + Yelp | VSFC (VI) | Đối nghịch hóa không gian vector |
| **S9** | **Unified Multi-task**| EN, FR, VI, Yelp | VSFC (VI) | Khung giải pháp hợp nhất đa nhiệm |
| **S10-S12**| **Ablation Study** | Tương đương S3, S4, S8 | Amazon/VSFC | So sánh XLM-R với mBERT |

---

## 5. Tiền xử lý (Preprocessing)
1.  **Chuyển đổi Binary:** Lọc bỏ nhãn Neutral (Trung tính) ở cả Amazon và VSFC để thống nhất bài toán Binary Classification (Tích cực / Tiêu cực).
2.  **Tokenization:** Sử dụng `XLM-RoBERTa Tokenizer` (hoặc `mBERT Tokenizer`) với `max_length=128`. Cả hai tokenizers này đều hỗ trợ đa ngôn ngữ nguyên bản.
3.  **Tối ưu hóa:** Dữ liệu hỗ trợ đọc trực tiếp từ định dạng CSV hoặc Parquet, đảm bảo tốc độ tải nhanh.
