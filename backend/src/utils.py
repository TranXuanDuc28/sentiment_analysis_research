import torch
import random
import numpy as np
import os
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Đã lưu checkpoint tại: {path}")

def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu kết quả nghiên cứu vào: {path}")

def print_dataset_statistics(labels, name):
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    print(f"\n📊 Dataset Statistics: {name}")
    print("-" * 45)
    print(f"{'Class':<15} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 45)
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for i in range(3):
        count = counts.get(i, 0)
        perc = (count / total) * 100 if total > 0 else 0
        print(f"{mapping[i]:<15} | {count:<10} | {perc:>10.2f}%")
    print("-" * 45)
    print(f"{'Total Samples':<15} | {total:<10} | {'100.00%':>10}")
    print("=" * 45)

def print_banner(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)
