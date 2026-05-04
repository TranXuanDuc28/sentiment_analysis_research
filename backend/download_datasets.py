
import os
import pandas as pd
import sys

# Configure stdout to use utf-8 to prevent UnicodeEncodeError with emojis on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def download():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # URL cấu trúc chuẩn của HuggingFace cho bản đã convert sang Parquet
    # Mẫu: .../resolve/refs%2Fconvert%2Fparquet/[CONFIG]/[SPLIT]/0000.parquet
    datasets = [
        {
            "name": "Amazon EN",
            "url_pattern": "https://huggingface.co/datasets/amazon_reviews_multi/resolve/refs%2Fconvert%2Fparquet/en/{split}/0000.parquet",
            "splits": ["train", "test", "validation"],
            "out_prefix": "amazon_en",
            "type": "parquet"
        },
        {
            "name": "VSFC",
            "url_pattern": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet",
            "splits": ["train", "test", "validation"],
            "out_prefix": "vsfc",
            "type": "parquet"
        },
        {
            "name": "Yelp",
            "url_pattern": "https://huggingface.co/datasets/yelp_polarity/resolve/refs%2Fconvert%2Fparquet/plain_text/{split}/0000.parquet",
            "splits": ["train", "test"],
            "out_prefix": "yelp",
            "type": "parquet"
        },
        {
            "name": "IMDB",
            "url_pattern": "https://huggingface.co/datasets/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/{split}/0000.parquet",
            "splits": ["train", "test"],
            "out_prefix": "imdb",
            "type": "parquet"
        }
    ]

    for ds in datasets:
        print(f"\n🚀 Đang tải {ds['name']}...")
        for split in ds["splits"]:
            try:
                url = ds["url_pattern"].format(split=split)
                if ds["type"] == "parquet":
                    df = pd.read_parquet(url)
                else:
                    df = pd.read_csv(url)
                
                # Chuẩn hóa tên cột cho VSFC nếu cần
                if ds["name"] == "VSFC":
                    if "sentence" in df.columns:
                        df = df.rename(columns={"sentence": "sentence", "sentiment": "sentiment"})
                
                # Giới hạn mẫu cho Yelp và IMDB (match SentXFormer paper)
                if ds["name"] in ["Yelp", "IMDB"] and len(df) > 8000:
                    df = df.sample(n=8000, random_state=42)
                    
                out_path = f"{data_dir}/{ds['out_prefix']}_{split}.csv"
                df.to_csv(out_path, index=False)
                print(f"✅ Đã lưu {out_path} ({len(df)} samples)")
            except Exception as e:
                print(f"❌ Lỗi khi tải {split}: {e}")

    print("\n✨ HOÀN THÀNH! Toàn bộ dữ liệu đã nằm trong thư mục 'data/'.")

if __name__ == "__main__":
    download()
