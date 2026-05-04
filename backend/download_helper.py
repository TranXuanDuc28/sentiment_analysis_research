
import os
import pandas as pd

# Tạo thư mục data nếu chưa có
os.makedirs("data", exist_ok=True)

def download_and_save():
    # Link "vàng" - Bản đã convert sang Parquet của HuggingFace
    vsfc_urls = {
        "train": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "test": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        "validation": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
    }

    # Link "vàng" cho TweetEval (Subset: Sentiment)
    twitter_urls = {
        "train": "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/refs%2Fconvert%2Fparquet/sentiment/train/0000.parquet",
        "test": "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/refs%2Fconvert%2Fparquet/sentiment/test/0000.parquet",
        "validation": "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/refs%2Fconvert%2Fparquet/sentiment/validation/0000.parquet"
    }

    print("🚀 Đang tải VSFC (Direct Link)...")
    for split, url in vsfc_urls.items():
        try:
            df = pd.read_parquet(url)
            df.to_csv(f"data/vsfc_{split}.csv", index=False)
            print(f"✅ Đã lưu data/vsfc_{split}.csv ({len(df)} samples)")
        except Exception as e:
            print(f"❌ Lỗi tải VSFC {split}: {e}")

    print("\n🚀 Đang tải TweetEval Sentiment (Direct Link)...")
    for split, url in twitter_urls.items():
        try:
            df = pd.read_parquet(url)
            df.to_parquet(f"data/twitter_{split}.parquet", index=False)
            print(f"✅ Đã lưu data/twitter_{split}.parquet ({len(df)} samples)")
        except Exception as e:
            print(f"❌ Lỗi tải Twitter {split}: {e}")

    print("\n✨ XONG! Toàn bộ dữ liệu đã sẵn sàng.")

if __name__ == "__main__":
    download_and_save()
