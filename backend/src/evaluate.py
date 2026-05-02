import torch
import numpy as np
import json
import os
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

def print_dataset_statistics(texts, labels, domain_name="Unknown"):
    """In bảng thống kê chi tiết về phân phối lớp của tập dữ liệu."""
    total = len(texts)
    counts = Counter(labels)
    
    print(f"\n📊 Dataset Statistics: {domain_name}")
    print("-" * 45)
    print(f"{'Class':<15} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 45)
    
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for label_id in sorted(label_names.keys()):
        count = counts.get(label_id, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{label_names[label_id]:<15} | {count:<10} | {percentage:>9.2f}%")
    
    print("-" * 45)
    print(f"{'Total Samples':<15} | {total:<10} | {'100.00%':>9}")
    print("=" * 45)

def evaluate_model(model, dataloader, device="cuda", scenario_name="Unknown"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {scenario_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            s_logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(s_logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "detailed_report": report
    }

    print(f"\n[Results: {scenario_name}]")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    return metrics

def save_research_results(results, save_path="results/research_results.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Đã lưu kết quả nghiên cứu vào: {save_path}")

def print_summary_table(results):
    print("\n" + "="*60)
    print(f"{'Scenario':<25} | {'Accuracy':<10} | {'F1-Macro':<10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<25} | {m['accuracy']:<10.4f} | {m['f1_macro']:<10.4f}")
    print("="*60)
