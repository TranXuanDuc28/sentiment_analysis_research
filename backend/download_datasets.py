
import os
import pandas as pd
from datasets import load_dataset

def download():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print("🚀 Đang tải bộ dữ liệu Amazon (MTEB version)...")
    # Tải Amazon English (Dùng cho S1, S2, S3, S4, S6)
    for split in ["train", "test", "validation"]:
        ds = load_dataset("mteb/amazon_reviews_multi", "en", split=split)
        df = ds.to_pandas()
        df.to_csv(f"{data_dir}/amazon_en_{split}.csv", index=False)
        print(f"✅ Đã lưu amazon_en_{split}.csv")

    print("\n🚀 Đang tải bộ dữ liệu VSFC (Vietnamese)...")
    # Tải VSFC (Dùng cho S5, S6)
    for split in ["train", "test", "validation"]:
        ds = load_dataset("uitnlp/vietnamese_students_feedback", split=split)
        df = ds.to_pandas()
        df.to_csv(f"{data_dir}/vsfc_{split}.csv", index=False)
        print(f"✅ Đã lưu vsfc_{split}.csv")

    print("\n🚀 Đang tải bộ dữ liệu TweetEval (Twitter)...")
    # Tải TweetEval (Dùng cho S4)
    for split in ["train", "test"]:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
        df = ds.to_pandas()
        df.to_csv(f"{data_dir}/twitter_{split}.csv", index=False)
        print(f"✅ Đã lưu twitter_{split}.csv")

    print("\n✨ TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC TẢI VỀ OFFLINE!")

if __name__ == "__main__":
    download()
