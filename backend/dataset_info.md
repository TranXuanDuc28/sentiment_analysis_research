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

| # | Nhóm Kịch bản | Dữ liệu Train (Source) | Dữ liệu Test (Target) | Khó khăn |
|---|---|---|---|---|
| **S1-S8** | **Multidomain (Chỉ Tiếng Anh)** | IMDb, Yelp | Amazon | Lệch miền (Domain Shift) |
| **S9** | **Domain SFT** | IMDb/Yelp + 500 Amazon | Amazon | Vượt qua lệch miền nhờ Fine-tuning |
| **S1b, S2, S3** | **Multilingual (Zero-shot & Joint)** | IMDb (EN), VSFC (VI) | VSFC (VI) | Lệch ngôn ngữ (Language Shift) |
| **S12** | **Language SFT** | IMDb (EN) + 500 VSFC | VSFC (VI) | Vượt qua rào cản ngôn ngữ nhờ Fine-tuning |
| **S13** | **Translation-Based** | Khởi tạo mô hình IMDb (EN) | VSFC (VI) đã dịch sang (EN) | Đánh giá chất lượng dịch máy (MT) |
| **S10, S14** | **Double Shift (Unified)** | IMDb + Yelp (EN) | VSFC (VI) | Lệch CẢ miền VÀ ngôn ngữ |

---

## 5. Tiền xử lý (Preprocessing)
1.  **Chuyển đổi Binary:** Lọc bỏ nhãn Neutral (Trung tính) ở cả Amazon và VSFC để thống nhất bài toán Binary Classification (Tích cực / Tiêu cực).
2.  **Tokenization:** Sử dụng `XLM-RoBERTa Tokenizer` (hoặc `mBERT Tokenizer`) với `max_length=128`. Cả hai tokenizers này đều hỗ trợ đa ngôn ngữ nguyên bản.
3.  **Tối ưu hóa:** Dữ liệu hỗ trợ đọc trực tiếp từ định dạng CSV hoặc Parquet, đảm bảo tốc độ tải nhanh.
