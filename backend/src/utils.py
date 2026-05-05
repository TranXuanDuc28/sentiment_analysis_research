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
    mapping = {0: "Negative", 1: "Positive"}
    for i in range(2):
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

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"🛑 Early stopping triggered after {self.counter} epochs without improvement.")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
