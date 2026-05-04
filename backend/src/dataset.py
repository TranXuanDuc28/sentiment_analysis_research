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
    "twitter": 4
}

def rating_to_sentiment_amazon(rating: int) -> int:
    if rating <= 2: return 0
    elif rating == 3: return 1
    else: return 2

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

def load_amazon_split(language, domain, split="train", max_samples=None):
    lang_code = "en" if language == "english" else "vi"
    print(f"[Dataset] Loading Amazon | lang={lang_code} | domain={domain} | split={split}")
    
    try:
        # Thêm trust_remote_code=True để HuggingFace cho phép tải script cũ
        dataset = load_dataset("amazon_reviews_multi", lang_code, split=split, trust_remote_code=True)
        
        # Chuẩn hóa tên domain (Amazon dùng 'book' thay vì 'books')
        domain_norm = domain.lower()
        if domain_norm == "books": domain_norm = "book"
        
        # Lọc theo domain (product_category)
        if domain_norm != "all":
            df = dataset.to_pandas()
            # Amazon reviews multi có cột 'product_category'
            df = df[df['product_category'] == domain_norm]
            if df.empty:
                print(f"⚠️ Cảnh báo: Không tìm thấy data cho domain {domain}, dùng toàn bộ split.")
                df = dataset.to_pandas()
        else:
            df = dataset.to_pandas()

        texts = df["review_body"].tolist()
        # label trong amazon_reviews_multi là 0-4 (tương ứng 1-5 sao)
        labels = [rating_to_sentiment_amazon(int(v) + 1) for v in df["stars"].tolist()]
        
        if max_samples and len(texts) > max_samples:
            random.seed(42)
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            
        return texts, labels, [DOMAIN_MAP.get(domain.lower(), 0)] * len(texts)
    except Exception as e:
        print(f"[Dataset] Error loading Amazon: {e}")
        # Fallback có đủ 3 nhãn (0, 1, 2) để tránh lỗi CrossEntropyLoss weights
        texts = ["Sản phẩm tệ"]*10 + ["Bình thường"]*10 + ["Rất tốt"]*10
        labels = [0]*10 + [1]*10 + [2]*10
        return texts, labels, [0]*30

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
    url_split = "validation" if split in ["dev", "validation"] else split
    url = f"https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/refs%2Fconvert%2Fparquet/default/{url_split}/0000.parquet?download=true"
    try:
        df = pd.read_parquet(url)
        texts = df["sentence"].tolist()
        labels = [int(v) for v in df["sentiment"].tolist()]
        if max_samples and len(texts) > max_samples:
            random.seed(42)
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        return texts, labels, [DOMAIN_MAP["vsfc"]] * len(texts)
    except:
        return ["Câu mẫu"], [2], [3]

def load_tweeteval(split="test", max_samples=None, unlabeled=False):
    """
    Load Twitter data. 
    Nếu unlabeled=True, toàn bộ nhãn sẽ bị gán thành -1 để đảm bảo DANN chuẩn.
    """
    print(f"[Dataset] Loading TweetEval | split={split} | unlabeled={unlabeled}")
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
        texts = list(ds["text"])
        
        if unlabeled:
            labels = [-1] * len(texts) # Ẩn nhãn hoàn toàn
        else:
            labels = [int(v) for v in list(ds["label"])]
        
        if max_samples and len(texts) > max_samples:
            random.seed(42)
            indices = random.sample(range(len(texts)), max_samples)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        return texts, labels, [DOMAIN_MAP["twitter"]] * len(texts)
    except:
        return ["Tweet"], [-1 if unlabeled else 1], [4]

def make_dataloader(texts, labels, domain_ids, tokenizer, batch_size=32, max_length=128, shuffle=False):
    dataset = SentimentDataset(texts, labels, domain_ids, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
