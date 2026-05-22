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
    
    label_names = {0: "Negative", 1: "Positive"}
    for label_id in sorted(label_names.keys()):
        count = counts.get(label_id, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{label_names[label_id]:<15} | {count:<10} | {percentage:>9.2f}%")
    
    print("-" * 45)
    print(f"{'Total Samples':<15} | {total:<10} | {'100.00%':>9}")
    print("=" * 45)

def evaluate_model(model, dataloader, device="cuda", scenario_name="Unknown"):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {scenario_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Robust unpacking: Take the first output (sentiment logits) regardless of how many values are returned
            outputs = model(input_ids, attention_mask)
            s_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            preds = torch.argmax(s_logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính toán chi tiết
    unique_labels = np.unique(all_labels)
    target_names = [str(i) for i in unique_labels]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    print(classification_report(all_labels, all_preds))
    
    # Macro F1 cực kỳ quan trọng cho dữ liệu không cân bằng (như VSFC)
    macro_f1 = report['macro avg']['f1-score']
    print(f"📊 Macro F1-score: {macro_f1:.4f}")
    
    # Vẽ Confusion Matrix (Lưu vào thư mục results/plots)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        SCENARIO_TITLES = {
            "S1_SingleSource_TL": "S1: Single-source TL",
            "S2_MD_Baseline": "S2: Multi-source Baseline",
            "S3_MD_FewShot": "S3: Few-shot Domain Adaptation",
            "S4_MD_MTL": "S4: Multi-task Learning",
            "S5_Multidomain_DANN": "S5: Multidomain DANN",
            "S6_Monolingual_VI": "S6: Monolingual VI Baseline",
            "S7_ML_ZeroShot": "S7: Multilingual Zero-shot",
            "S8_ML_FewShot": "S8: Multilingual Few-shot",
            "S9_ML_Translation": "S9: Translation-based Method",
            "S10_UN_FewShot": "S10: Unified Few-shot Target Fine-Tuning",
            "S11_UN_ZeroShot": "S11: Unified Zero-shot",
            "S12_UN_DANN": "S12: Unified DANN",
            "S13_UN_MultiTask": "S13: Unified Multi-task",
            "S14_mBERT_MD": "S14: mBERT MD DANN",
            "S15_mBERT_ML": "S15: mBERT ML Zero-shot",
            "S16_mBERT_UN": "S16: mBERT UN DANN"
        }
        clean_title = SCENARIO_TITLES.get(scenario_name, scenario_name)
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {clean_title}')
        
        os.makedirs("results/plots", exist_ok=True)
        plot_path = f"results/plots/cm_{scenario_name.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Đã lưu Confusion Matrix tại: {plot_path}")
    except Exception as e:
        print(f"⚠️ Không thể vẽ Confusion Matrix: {e}")

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "detailed_report": report
    }

    print(f"\n[Results: {scenario_name}]")
    
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
