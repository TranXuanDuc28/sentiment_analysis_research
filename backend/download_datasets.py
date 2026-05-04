
import os
import pandas as pd

def download():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Mapping các URL trực tiếp (Bỏ qua load_dataset script)
    datasets = {
        "Amazon EN": {
            "base": "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/main/en",
            "files": ["train", "test", "validation"],
            "type": "parquet",
            "out_prefix": "amazon_en"
        },
        "VSFC": {
            "base": "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/main/data",
            "files": ["train", "test", "validation"],
            "type": "csv",
            "out_prefix": "vsfc"
        },
        "TweetEval": {
            "base": "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/main/sentiment",
            "files": ["train", "test", "validation"],
            "type": "parquet",
            "out_prefix": "twitter"
        }
    }

    for name, info in datasets.items():
        print(f"\n🚀 Đang tải {name}...")
        for f in info["files"]:
            try:
                if info["type"] == "parquet":
                    # Thử các pattern phổ biến của HF
                    url = f"{info['base']}/{f}.parquet"
                    try:
                        df = pd.read_parquet(url)
                    except:
                        url = f"{info['base']}/{f}-00000-of-00001.parquet"
                        df = pd.read_parquet(url)
                else:
                    url = f"{info['base']}/{f}.csv"
                    df = pd.read_csv(url)
                
                out_path = f"{data_dir}/{info['out_prefix']}_{f}.csv"
                df.to_csv(out_path, index=False)
                print(f"✅ Đã lưu {out_path}")
            except Exception as e:
                print(f"❌ Lỗi khi tải {f}: {e}")

    print("\n✨ HOÀN THÀNH! Dữ liệu đã sẵn sàng trong thư mục 'data/'.")

if __name__ == "__main__":
    download()
