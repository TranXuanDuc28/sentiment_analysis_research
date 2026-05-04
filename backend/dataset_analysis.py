
import pandas as pd
import os
from src.dataset import load_amazon_split, load_vsfc, load_tweeteval

def get_distribution(labels):
    s = pd.Series(labels).value_counts().sort_index()
    dist = {0: 0, 1: 0, 2: 0}
    for k, v in s.items():
        dist[k] = v
    return dist

def run_analysis():
    print("\n" + "="*60)
    print("      PHÂN TÍCH THỐNG KÊ BỘ DỮ LIỆU NGHIÊN CỨU")
    print("="*60)
    
    results = []

    # 1. Amazon Domains
    for domain in ["books", "electronics", "apparel"]:
        try:
            t, l, _ = load_amazon_split("english", domain, "train")
            d = get_distribution(l)
            results.append(["Amazon " + domain.capitalize(), len(t), d[0], d[1], d[2]])
        except: pass

    # 2. VSFC (Vietnamese)
    try:
        t, l, _ = load_vsfc("train")
        d = get_distribution(l)
        results.append(["UIT-VSFC (VI)", len(t), d[0], d[1], d[2]])
    except: pass

    # 3. Twitter (TweetEval)
    try:
        t, l, _ = load_tweeteval("train")
        d = get_distribution(l)
        results.append(["TweetEval (EN)", len(t), d[0], d[1], d[2]])
    except: pass

    # In bảng kết quả
    df = pd.DataFrame(results, columns=["Tên Bộ Dữ Liệu", "Tổng số mẫu", "Tiêu cực (0)", "Trung tính (1)", "Tích cực (2)"])
    print(df.to_string(index=False))
    print("="*60)
    print("Ghi chú: 0=Negative, 1=Neutral, 2=Positive")

if __name__ == "__main__":
    run_analysis()
