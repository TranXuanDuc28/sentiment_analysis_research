import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCENARIOS = {
    "s1": {"file": "results_s1.json", "name": "S1_SingleSource_TL", "title": "S1: Single-source TL"},
    "s2": {"file": "results_s2.json", "name": "S2_MD_Baseline", "title": "S2: Multi-source Baseline"},
    "s3": {"file": "results_s3.json", "name": "S3_MD_FewShot", "title": "S3: Few-shot Domain Adaptation"},
    "s4": {"file": "results_s4.json", "name": "S4_MD_MTL", "title": "S4: Multi-task Learning"},
    "s5": {"file": "results_s5.json", "name": "S5_Multidomain_DANN", "title": "S5: Multidomain DANN"},
    "s6": {"file": "results_s6.json", "name": "S6_Monolingual_VI", "title": "S6: Monolingual VI Baseline"},
    "s7": {"file": "results_s7.json", "name": "S7_ML_ZeroShot", "title": "S7: Multilingual Zero-shot"},
    "s8": {"file": "results_s8.json", "name": "S8_ML_FewShot", "title": "S8: Multilingual Few-shot"},
    "s9": {"file": "results_s9.json", "name": "S9_ML_Translation", "title": "S9: Translation-based Method"},
    "s10": {"file": "results_s10.json", "name": "S10_UN_FewShot", "title": "S10: Unified Few-shot Target Fine-Tuning"},
    "s11": {"file": "results_s11.json", "name": "S11_UN_ZeroShot", "title": "S11: Unified Zero-shot"},
    "s12": {"file": "results_s12.json", "name": "S12_UN_DANN", "title": "S12: Unified DANN"},
    "s13": {"file": "results_s13.json", "name": "S13_UN_MultiTask", "title": "S13: Unified Multi-task"},
    "s14": {"file": "results_s14.json", "name": "S14_mBERT_MD", "title": "S14: mBERT MD DANN"},
    "s15": {"file": "results_s15.json", "name": "S15_mBERT_ML", "title": "S15: mBERT ML Zero-shot"},
    "s16": {"file": "results_s16.json", "name": "S16_mBERT_UN", "title": "S16: mBERT UN DANN"}
}

results_dir = r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\project\backend\results"
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Delete old joint training confusion matrix if it exists
old_cm = os.path.join(plots_dir, "cm_s10_ml_joint.png")
if os.path.exists(old_cm):
    os.remove(old_cm)
    print(f"[CLEANUP] Deleted old unused plot: {old_cm}")

for s_id, s_info in SCENARIOS.items():
    json_path = os.path.join(results_dir, s_info["file"])
    if not os.path.exists(json_path):
        print(f"[WARN] Result file {json_path} not found. Skipping.")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        report = data["detailed_report"]
        # Reconstruct confusion matrix cells from recall and support
        # Class 0: Negative
        support_0 = float(report["0"]["support"])
        recall_0 = float(report["0"]["recall"])
        tn = int(round(recall_0 * support_0))
        fp = int(support_0 - tn)

        # Class 1: Positive
        support_1 = float(report["1"]["support"])
        recall_1 = float(report["1"]["recall"])
        tp = int(round(recall_1 * support_1))
        fn = int(support_1 - tp)

        cm = np.array([[tn, fp], [fn, tp]])
        target_names = ["0", "1"]

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {s_info["title"]}')

        dest_file = f"cm_{s_info['name'].lower().replace(' ', '_')}.png"
        dest_path = os.path.join(plots_dir, dest_file)
        plt.savefig(dest_path)
        plt.close()
        print(f"[OK] Rebuilt confusion matrix plot for {s_info['title']} -> {dest_file}")
    except Exception as e:
        print(f"[ERROR] Failed to rebuild plot for {s_info['title']}: {e}")

print("[INFO] Rebuilding confusion matrices completed successfully!")
