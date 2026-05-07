import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Lộ trình 11 kịch bản: Tập trung thẳng vào Multidomain và Multilingual
PHASES = {
    "Chặng 1: Multidomain (RQ1)": ["S1", "S2", "S3"],
    "Chặng 2: Multilingual (RQ2)": ["S4", "S5", "S6"],
    "Chặng 3: Unified Framework (RQ3)": ["S7", "S8", "S9"],
    "Chặng 4: Model Ablation": ["S3", "S10", "S8", "S11"]
}

LABELS = {
    "S1": "MD Baseline (Merge)",
    "S2": "MD Multi-task",
    "S3": "MD DANN (XLM-R)",
    "S4": "ML Zero-shot",
    "S5": "ML Translation",
    "S6": "ML Joint",
    "S7": "Unified Zero-shot",
    "S8": "Unified DANN (XLM-R)",
    "S9": "Unified Multi-task",
    "S10": "MD DANN (mBERT)",
    "S11": "Unified DANN (mBERT)"
}

def generate_aggregate_report(results_dir="results", plots_dir="results/plots"):
    print("📊 Đang khởi tạo báo cáo tổng hợp (Bắt đầu từ Multidomain)...")
    os.makedirs(plots_dir, exist_ok=True)
    
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith(".json") and filename.startswith("results_s"):
            path = os.path.join(results_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    key_id = filename.replace("results_", "").replace(".json", "").upper()
                    results[key_id] = data.get("f1_macro", 0)
            except:
                continue

    if not results:
        print("⚠️ Không tìm thấy file kết quả nào.")
        return

    sns.set_style("whitegrid")
    for phase_name, scenario_ids in PHASES.items():
        data_list = []
        for sid in scenario_ids:
            if sid in results:
                data_list.append({"Scenario": sid, "F1-Macro": results[sid], "Technique": LABELS.get(sid, sid)})
        
        if data_list:
            df_phase = pd.DataFrame(data_list)
            plt.figure(figsize=(10, 6))
            palette = "viridis" if "Chặng 4" not in phase_name else "coolwarm"
            ax = sns.barplot(x="Scenario", y="F1-Macro", data=df_phase, palette=palette, hue=None)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=11, color='black', xytext=(0, 8),
                            textcoords='offset points', fontweight='bold')
            plt.title(f"{phase_name}\nPerformance Analysis", fontsize=14, fontweight='bold')
            plt.ylim(0.0, 1.0)
            safe_name = phase_name.split(":")[0].replace(" ", "_").lower()
            plt.savefig(os.path.join(plots_dir, f"report_{safe_name}.png"), bbox_inches='tight', dpi=300)
            plt.close()

    summary_path = os.path.join(results_dir, "research_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Kết quả (Lộ trình 11 Kịch bản: Đa miền & Đa ngôn ngữ)\n\n")
        for phase_name, scenario_ids in PHASES.items():
            f.write(f"### {phase_name}\n")
            f.write("| Mã | Kỹ thuật | F1-Macro |\n| :--- | :--- | :--- |\n")
            for sid in scenario_ids:
                if sid in results:
                    f.write(f"| **{sid}** | {LABELS.get(sid, '-')} | {results[sid]:.4f} |\n")
            f.write("\n")
    print(f"🎉 Hoàn tất báo cáo tại '{summary_path}'")

if __name__ == "__main__":
    generate_aggregate_report()
