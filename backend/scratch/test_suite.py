import requests
import json
import sys

# Đảm bảo in ký tự UTF-8 được hiển thị đúng trên Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://localhost:8000/api"

TEST_CASES = [
    {
        "description": "1. Câu tích cực có chứa từ khóa khía cạnh (Price và Quality)",
        "text": "Sản phẩm chất lượng cực kỳ tốt, giá cả lại rất rẻ và phải chăng.",
        "endpoint": "predict"
    },
    {
        "description": "2. Câu tiêu cực có chứa từ khóa khía cạnh (Service và Delivery)",
        "text": "Giao hàng quá chậm trễ, đóng gói sơ sài và thái độ phục vụ của nhân viên tư vấn rất tệ.",
        "endpoint": "predict"
    },
    {
        "description": "3. Câu trung tính",
        "text": "Sách được đóng gói bằng bìa các-tông bình thường.",
        "endpoint": "predict"
    },
    {
        "description": "4. Phân tích độ lệch miền (Domain Shift) - câu thuộc miền sách tiếng Anh",
        "text": "The plot twist at the end of the novel was absolutely thrilling and well written.",
        "endpoint": "domain-analysis"
    },
    {
        "description": "5. Phân tích độ lệch miền (Domain Shift) - câu lệch miền (tiếng Việt)",
        "text": "Tôi muốn mua một chiếc điện thoại di động giá rẻ.",
        "endpoint": "domain-analysis"
    }
]

def main():
    print("="*60)
    print("   BỘ CHƯƠNG TRÌNH KIỂM THỬ ĐỒNG BỘ FRONTEND - BACKEND")
    print("="*60)
    
    # Kiểm tra xem server đang chạy không
    try:
        requests.get(f"{BASE_URL}/history", timeout=3)
    except requests.exceptions.ConnectionError:
        print("❌ Lỗi: Server Backend chưa chạy tại http://localhost:8000.")
        print("👉 Vui lòng chạy lệnh: python api/main.py trước khi chạy file test này.")
        sys.exit(1)
        
    for case in TEST_CASES:
        desc = case["description"]
        text = case["text"]
        endpoint = case["endpoint"]
        
        print(f"\n★ Test case: {desc}")
        print(f"  Đầu vào: \"{text}\"")
        
        url = f"{BASE_URL}/{endpoint}"
        try:
            response = requests.post(url, json={"text": text})
            if response.status_code == 200:
                data = response.json()
                print("  [Kết quả trả về]:")
                if endpoint == "predict":
                    print(f"    - Cảm xúc phân loại (Sentiment): {data['sentiment']} (Độ tin cậy: {data['confidence'] * 100}%)")
                    
                    aspects = data.get("aspects", [])
                    if aspects:
                        print("    - Khía cạnh trích xuất (Aspects):")
                        for a in aspects:
                            print(f"       + {a['aspect']}: {a['sentiment']}")
                    else:
                        print("    - Khía cạnh trích xuất: Không tìm thấy khía cạnh nào.")
                        
                    # Hiển thị độ quan trọng của từ
                    explanation = data.get("explanation", [])
                    if explanation:
                        # Lọc lấy top 5 từ quan trọng nhất
                        top_words = sorted(explanation, key=lambda x: x["score"], reverse=True)[:5]
                        print("    - Top 5 từ quan trọng nhất (Word Importance):")
                        for w in top_words:
                            print(f"       + '{w['word']}': {w['score']}")
                            
                elif endpoint == "domain-analysis":
                    print(f"    - Độ tương đồng với miền huấn luyện (Similarity Score): {data['similarity_score'] * 100}%")
                    print(f"    - Độ lệch miền (Shift Magnitude): {data['shift_magnitude']}")
                    print(f"    - Đánh giá (Status): {data['status']}")
                    print(f"    - Tọa độ biểu đồ SVG 2D: TargetPoint: {data['target_point']}")
            else:
                print(f"  ❌ Lỗi API (Status Code: {response.status_code}): {response.text}")
        except Exception as e:
            print(f"  ❌ Gặp lỗi kết nối API: {e}")
            
    print("\n" + "="*60)
    # Lấy lịch sử
    try:
        history_resp = requests.get(f"{BASE_URL}/history")
        if history_resp.status_code == 200:
            hist = history_resp.json()
            print(f"✅ Kiểm thử lịch sử thành công! Tổng số bản ghi trong lịch sử: {len(hist)}")
        else:
            print("❌ Lỗi khi lấy lịch sử hoạt động.")
    except Exception as e:
        print(f"❌ Lỗi khi lấy lịch sử: {e}")
    print("="*60)

if __name__ == "__main__":
    main()
