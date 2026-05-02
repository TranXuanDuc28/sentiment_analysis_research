# Dataset Description: Cross-Domain & Cross-Lingual Sentiment Analysis

Tài liệu này mô tả chi tiết các tập dữ liệu được sử dụng trong hệ thống phân tích cảm xúc đa miền và đa ngôn ngữ.

---

## 1. Amazon Reviews Multi (MARC)
Đây là tập dữ liệu chính được sử dụng để huấn luyện (Source domain) và đánh giá khả năng chuyển đổi ngôn ngữ/miền (Target domain).

*   **Nguồn:** [HuggingFace - amazon_reviews_multi](https://huggingface.co/datasets/amazon_reviews_multi)
*   **Ngôn ngữ sử dụng:** 
    *   **English (EN):** Dùng để huấn luyện mô hình cơ sở.
    *   **Vietnamese (VI):** Dùng để đánh giá khả năng chuyển đổi ngôn ngữ (Cross-lingual).
*   **Các miền (Domains):** 
    *   `Books` (Miền chính để huấn luyện).
    *   `Electronics` (Dùng để đánh giá Domain Shift).
*   **Cấu trúc dữ liệu:**
    *   `review_body`: Nội dung đánh giá.
    *   `review_title`: Tiêu đề đánh giá (được nối với body trong quá trình tiền xử lý).
    *   `stars`: Xếp hạng từ 1 đến 5 sao.
*   **Ánh xạ nhãn (Label Mapping):**
    *   `1 - 2 stars`: 0 (Negative)
    *   `3 stars`: 1 (Neutral)
    *   `4 - 5 stars`: 2 (Positive)

---

## 2. UIT-VSFC (Vietnamese Students' Feedback Corpus)
Tập dữ liệu dùng để đánh giá khả năng tổng quát hóa trên một tập dữ liệu hoàn toàn mới và khác biệt về miền (Cross-dataset).

*   **Nguồn:** [HuggingFace - uitnlp/vietnamese_students_feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)
*   **Ngôn ngữ:** Tiếng Việt (VI).
*   **Miền (Domain):** Giáo dục / Phản hồi của sinh viên (khác hoàn toàn với miền thương mại điện tử của Amazon).
*   **Cấu trúc dữ liệu:**
    *   `sentence`: Câu phản hồi của sinh viên.
    *   `sentiment`: Nhãn cảm xúc (0, 1, 2).
*   **Ánh xạ nhãn:**
    *   `0`: Negative
    *   `1`: Neutral
    *   `2`: Positive

---

## 3. Phân bổ dữ liệu trong các kịch bản thực nghiệm

| # | Kịch bản | Tập dữ liệu | Ngôn ngữ | Miền (Domain) | Mục đích |
|---|---|---|---|---|---|
| **1** | **In-domain** | MARC | English | Books | Thiết lập baseline (hiệu năng tối ưu nhất). |
| **2** | **Domain Shift** | MARC | English | Electronics | Kiểm tra sự sụt giảm khi thay đổi miền (cùng ngôn ngữ). |
| **3** | **Language Shift** | MARC | Vietnamese | General | Kiểm tra khả năng hiểu tiếng Việt (cùng nguồn dữ liệu). |
| **4** | **Double Shift** | MARC | Vietnamese | Electronics | Thách thức kết hợp cả thay đổi miền và ngôn ngữ. |
| **5** | **Cross-Dataset** | UIT-VSFC | Vietnamese | Education | Kiểm tra khả năng tổng quát hóa thực tế trên tập dữ liệu mới. |

---

## 4. Thống kê số lượng mẫu (Configurable)
Hệ thống hỗ trợ cấu hình số lượng mẫu thông qua tham số `--quick` trong `main.py`:

*   **Chế độ FULL (Nghiên cứu):**
    *   Train: 2000 mẫu.
    *   Validation: 500 mẫu.
    *   Test: 500 mẫu cho mỗi kịch bản.
*   **Chế độ QUICK (Kiểm thử nhanh):**
    *   Train: 300 mẫu.
    *   Validation: 100 mẫu.
    *   Test: 150 mẫu.

---

## 5. Tiền xử lý (Preprocessing)
1.  **Nối văn bản:** Tiêu đề và nội dung review được nối lại để tối đa hóa thông tin.
2.  **Tokenization:** Sử dụng `XLM-RoBERTa Tokenizer` với `max_length=128`.
3.  **Tối ưu hóa:** Dữ liệu được load trực tiếp từ định dạng Parquet để tăng tốc độ và tránh lỗi script của HuggingFace.
