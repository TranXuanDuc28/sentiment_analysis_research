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

## 4. Phân bổ Dữ liệu trong các Kịch bản Thực nghiệm (16 Scenarios)

| Mã | Kỹ thuật (Theo slide bài giảng) | Dữ liệu Nguồn | Dữ liệu Đích | Thách thức / Phương pháp |
|---|---|---|---|---|
| **S1** | **Single-source Pretrained model-based TL** | IMDb (EN) | Amazon (EN) | Chuyển giao đơn nguồn lên miền mới |
| **S2** | **Multi-source Transfer Learning (without Adaptation)** | IMDb, Yelp (EN) | Amazon (EN) | Chuyển giao đa nguồn chưa thích ứng |
| **S3** | **Fine-tuning based Domain Adaptation** | model S2 + 500 Amazon | Amazon (EN) | Thích ứng miền dùng Few-shot |
| **S4** | **Multi-task Learning (Hard Parameter Sharing)** | IMDb, Yelp (EN) | IMDb, Yelp (EN) | Học song song nhiều miền |
| **S5** | **Feature-based Domain Adaptation (DANN)** | IMDb, Yelp (EN) | Amazon (EN) | Căn chỉnh vector phân phối bằng GRL |
| **S6** | **Monolingual VI Baseline** | VSFC (VI) | VSFC (VI) | Đánh giá nội bộ đơn ngữ |
| **S7** | **Cross-lingual TL based on Multilingual Models** | IMDb (EN), Amazon (FR) | VSFC (VI) | Lệch ngôn ngữ chéo (Zero-shot) |
| **S8** | **Cross-lingual Fine-tuning for Target Language** | model_src + 500 VSFC | VSFC (VI) | Cải tiến chéo ngôn ngữ dùng Few-shot |
| **S9** | **Translation-Based Method** | IMDb (EN), Amazon (FR) | VSFC -> EN | Dịch máy + Phân tích cảm xúc |
| **S10**| **Unified Few-shot Target Fine-Tuning** | model S2 + 500 VSFC | VSFC (VI) | Tinh chỉnh thích ứng đồng thời Miền & Ngôn ngữ |
| **S11**| **Unified Cross-lingual Domain Adaptation (Zero-shot)** | model S2 (IMDb + Yelp) | VSFC (VI) | Lệch đồng thời cả Miền & Ngôn ngữ (Zero-shot) |
| **S12**| **Unified Feature-based Domain Adaptation & Cross-lingual Transfer (DANN)** | IMDb, Yelp (EN), Amazon (FR) | VSFC (VI) | Căn chỉnh đối nghịch miền & ngôn ngữ (DANN) |
| **S13**| **Unified Multi-task Learning (Hard Parameter Sharing)** | IMDb, Yelp (EN), Amazon (FR), VSFC (VI) | VSFC (VI) | Giải pháp hợp nhất đa nhiệm |
| **S14**| **mBERT Feature-based Domain Adaptation** | IMDb, Yelp (EN) | Amazon (EN) | Đánh giá mBERT tương đương S5 |
| **S15**| **mBERT Cross-lingual TL based on Multilingual Models** | IMDb (EN), Amazon (FR) | VSFC (VI) | Đánh giá mBERT tương đương S7 |
| **S16**| **mBERT Unified Feature-based Domain Adaptation & Cross-lingual Transfer** | IMDb, Yelp (EN), Amazon (FR) | VSFC (VI) | Đánh giá mBERT tương đương S12 |

---

## 5. Tiền xử lý (Preprocessing)
1.  **Chuyển đổi Binary:** Lọc bỏ nhãn Neutral (Trung tính) ở cả Amazon và VSFC để thống nhất bài toán Binary Classification (Tích cực / Tiêu cực).
2.  **Tokenization:** Sử dụng `XLM-RoBERTa Tokenizer` (hoặc `mBERT Tokenizer`) với `max_length=128`. Cả hai tokenizers này đều hỗ trợ đa ngôn ngữ nguyên bản.
3.  **Tối ưu hóa:** Dữ liệu hỗ trợ đọc trực tiếp từ định dạng CSV hoặc Parquet, đảm bảo tốc độ tải nhanh.
