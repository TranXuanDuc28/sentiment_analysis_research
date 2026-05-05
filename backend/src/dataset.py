"""
dataset.py - V8 (Academic Correctness Version)
----------------------------------------------
- Hỗ trợ Multi-domain (Books, Electronics, Apparel).
- Hỗ trợ Unlabeled Data (Twitter train labels = -1).
- Ép kiểu list chuẩn và Domain ID chuẩn (0-4).
"""

import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Domain Mapping chuẩn nghiên cứu
DOMAIN_MAP = {
    "books": 0,
    "electronics": 1,
    "apparel": 2,
    "vsfc": 3,
    "yelp": 4,
    "imdb": 5
}

def rating_to_sentiment_amazon(rating: int) -> int:
    # Not used anymore since we map inline, but keeping signature if imported elsewhere
    return 0 if rating <= 2 else 1

def word_segment_vietnamese(texts):
    """Tách từ tiếng Việt cho PhoBERT"""
    try:
        from underthesea import word_tokenize
        return [word_tokenize(t, format="text") for t in texts]
    except ImportError:
        return texts

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, domain_ids, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.domain_ids = domain_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, max_length=self.max_length, 
                                  padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "domain_ids": torch.tensor(int(self.domain_ids[idx]), dtype=torch.long)
        }

def load_amazon_split(language, domain, split="train", max_samples=None, unlabeled=False):
    lang_code = "en" if language == "english" else "vi"
    # Kiểm tra linh hoạt các đường dẫn file
    possible_paths = [
        f"data/amazon_{lang_code}_{split}.parquet",
        f"data/amazon_{lang_code}_{split}.csv",
        f"data/amazon_{split}.csv", # Khớp với amazon_train.csv của bạn
        f"data/amazon/{split}.csv"  # Khớp với folder amazon/
    ]
    
    df = None
    for p in possible_paths:
        if os.path.exists(p):
            print(f"[Dataset] Found Amazon at: {p}")
            if p.endswith(".parquet"):
                df = pd.read_parquet(p)
            else:
                try:
                    df = pd.read_csv(p)
                    # Check columns to ensure it's not a headerless CSV
                    if "review_body" not in df.columns and "text" not in df.columns:
                        df = pd.read_csv(p, header=None, names=["idx", "review_id", "product_id", "reviewer_id", "stars", "review_body", "review_title", "language", "product_category"])
                except:
                    df = pd.read_csv(p, header=None, names=["idx", "review_id", "product_id", "reviewer_id", "stars", "review_body", "review_title", "language", "product_category"])
            break

    try:
        if df is None:
            print(f"⚠️ Không thấy file cục bộ, đang tải online...")
            dataset = load_dataset("mteb/amazon_reviews_multi", lang_code, split=split, trust_remote_code=True)
            df = dataset.to_pandas()

        # Tự động chọn tên cột (Hỗ trợ cả bản MTEB và bản Amazon gốc)
        text_col = "text" if "text" in df.columns else "review_body"
        label_col = "label" if "label" in df.columns else "stars"
        
        # Lọc theo ngôn ngữ nếu có cột 'language' (MARC gốc thường có cột này)
        if "language" in df.columns:
            df = df[df["language"] == lang_code]
            if df.empty:
                print(f"⚠️ Cảnh báo: Không tìm thấy dữ liệu cho ngôn ngữ '{lang_code}' trong file.")
        
        # Lọc theo lĩnh vực (Domain) - QUAN TRỌNG cho S1, S2, S3
        if domain.lower() != "all":
            # Chuẩn hóa tên domain (Amazon dùng 'book', 'electronics'...)
            domain_norm = domain.lower()
            if domain_norm == "books": domain_norm = "book"
            
            if "product_category" in df.columns:
                df = df[df["product_category"] == domain_norm]
                if df.empty:
                    print(f"⚠️ Cảnh báo: Không tìm thấy dữ liệu cho domain '{domain_norm}' trong file.")
            else:
                print(f"⚠️ Cảnh báo: File không có cột 'product_category', không thể lọc domain '{domain}'.")
        
        # Filter out Neutral to make it Binary (0 = Negative, 1 = Positive)
        if label_col == "label":
            # 0,1 are Negative; 2 is Neutral; 3,4 are Positive
            df = df[~df[label_col].isin([2, "2", 2.0])]
            labels = [0 if int(v) < 2 else 1 for v in df[label_col].tolist()]
        else:
            # 1,2 are Negative; 3 is Neutral; 4,5 are Positive
            df = df[~df[label_col].isin([3, "3", 3.0])]
            labels = [0 if int(v) < 3 else 1 for v in df[label_col].tolist()]
        
        texts = df[text_col].tolist()
        
        if unlabeled:
            labels = [-1] * len(texts)
            
        if max_samples and len(texts) > max_samples:
            # Seed khác nhau cho domain khác cho đa dạng
            d_seed = 42 + DOMAIN_MAP.get(domain.lower(), 0)
            random.seed(d_seed)
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            
        return texts, labels, [DOMAIN_MAP.get(domain.lower(), 0)] * len(texts)
    except Exception as e:
        print(f"[Dataset] Error: {e}")
        # Fallback 3 nhãn
        texts = ["Tệ"]*10 + ["Ổn"]*10 + ["Tốt"]*10
        return texts, [0]*10 + [1]*10 + [2]*10, [0]*30

def load_multi_domain_amazon(domains=["books", "electronics", "apparel"], max_samples=1000):
    """Gộp nhiều domain Amazon lại với nhau."""
    all_texts, all_labels, all_d_ids = [], [], []
    samples_per_domain = max_samples // len(domains)
    
    for d in domains:
        t, l, d_ids = load_amazon_split("english", d, "train", max_samples=samples_per_domain)
        all_texts.extend(t)
        all_labels.extend(l)
        all_d_ids.extend(d_ids)
    
    return all_texts, all_labels, all_d_ids

def load_vsfc(split="test", max_samples=None):
    csv_path = f"data/vsfc_{split}.csv"
    pq_path = f"data/vsfc_{split}.parquet"
    try:
        if os.path.exists(pq_path):
            df = pd.read_parquet(pq_path)
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            ds = load_dataset("uitnlp/vietnamese_students_feedback", split=split)
            df = ds.to_pandas()
            
        # Filter out Neutral (1)
        df = df[~df["sentiment"].isin([1, "1", 1.0])]
        texts = df["sentence"].tolist()
        labels = [1 if int(v) == 2 else 0 for v in df["sentiment"].tolist()]
        if max_samples and len(texts) > max_samples:
            random.seed(42)
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        return texts, labels, [DOMAIN_MAP["vsfc"]] * len(texts)
    except:
        return ["Câu mẫu"], [2], [3]

def load_yelp(split="test", max_samples=None, unlabeled=False):
    csv_path = f"data/yelp_{split}.csv"
    print(f"[Dataset] Loading Yelp | split={split} | unlabeled={unlabeled}")
    try:
        df = pd.read_csv(csv_path)
        texts = df["text"].tolist()
        
        if unlabeled:
            labels = [-1] * len(texts)
        else:
            # Map binary (0,1) to binary (0,1)
            labels = [1 if int(v) == 1 else 0 for v in df["label"].tolist()]
        
        if max_samples and len(texts) > max_samples:
            random.seed(42)
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        return texts, labels, [DOMAIN_MAP["yelp"]] * len(texts)
    except Exception as e:
        print(f"Error loading Yelp: {e}")
        return ["Yelp review"], [-1 if unlabeled else 2], [4]

def load_imdb(split="test", max_samples=None, unlabeled=False):
    csv_path = f"data/imdb_{split}.csv"
    print(f"[Dataset] Loading IMDB | split={split} | unlabeled={unlabeled}")
    try:
        df = pd.read_csv(csv_path)
        texts = df["text"].tolist()
        
        if unlabeled:
            labels = [-1] * len(texts)
        else:
            # Map binary (0,1) to binary (0,1)
            labels = [1 if int(v) == 1 else 0 for v in df["label"].tolist()]
        
        if max_samples and len(texts) > max_samples:
            random.seed(42)
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        return texts, labels, [DOMAIN_MAP["imdb"]] * len(texts)
    except Exception as e:
        print(f"Error loading IMDB: {e}")
        return ["IMDB review"], [-1 if unlabeled else 2], [5]

def make_dataloader(texts, labels, domain_ids, tokenizer, batch_size=32, max_length=128, shuffle=False):
    dataset = SentimentDataset(texts, labels, domain_ids, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
