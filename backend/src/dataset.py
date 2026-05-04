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
    file_path = f"data/amazon_{lang_code}_{split}.csv"
    
    print(f"[Dataset] Loading Amazon | file={file_path} | domain={domain}")
    
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            print(f"⚠️ Không thấy file cục bộ, đang tải online...")
            dataset = load_dataset("mteb/amazon_reviews_multi", lang_code, split=split)
            df = dataset.to_pandas()

        # Tự động chọn tên cột (Hỗ trợ cả bản MTEB và bản Amazon gốc)
        text_col = "text" if "text" in df.columns else "review_body"
        label_col = "label" if "label" in df.columns else "stars"
        
        texts = df[text_col].tolist()
        # Nếu là 'label' (0-4) thì +1, nếu là 'stars' (1-5) thì giữ nguyên để đưa vào hàm rating_to_sentiment_amazon
        labels = [rating_to_sentiment_amazon(int(v) + (1 if label_col == "label" else 0)) for v in df[label_col].tolist()]
        
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
    file_path = f"data/vsfc_{split}.csv"
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            ds = load_dataset("uitnlp/vietnamese_students_feedback", split=split)
            df = ds.to_pandas()
            
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
    file_path = f"data/twitter_{split}.csv"
    print(f"[Dataset] Loading Twitter | file={file_path} | unlabeled={unlabeled}")
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
            df = ds.to_pandas()
            
        texts = df["text"].tolist()
        
        if unlabeled:
            labels = [-1] * len(texts)
        else:
            labels = [int(v) for v in df["label"].tolist()]
        
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
