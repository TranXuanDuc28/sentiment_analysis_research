import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Lộ trình kịch bản: Đối sánh toàn diện (MD, ML, UN, Ablation)
PHASES = {
    "Chặng 1: Multidomain (RQ1)": ["S1", "S1A", "S1B", "S2", "S3"],
    "Chặng 2: Multilingual (RQ2)": ["S0", "S4", "S4B", "S5", "S6"],
    "Chặng 3: Unified Framework (RQ3)": ["S7", "S8", "S9"],
    "Chặng 4: Model Ablation (mBERT vs XLM-R)": ["S3", "S10", "S4", "S11", "S8", "S12"]
}

LABELS = {
    "S0": "Mono VI Baseline",
    "S1": "MD Baseline",
    "S1A": "MD Single-source",
    "S1B": "MD Few-shot",
    "S2": "MD Multi-task",
    "S3": "MD DANN (XLM-R)",
    "S4": "ML Zero-shot (XLM-R)",
    "S4B": "ML Few-shot",
    "S5": "ML Translation",
    "S6": "ML Joint",
    "S7": "Unified Zero-shot",
    "S8": "Unified DANN (XLM-R)",
    "S9": "Unified Multi-task",
    "S10": "MD DANN (mBERT)",
    "S11": "ML Zero-shot (mBERT)",
    "S12": "Unified DANN (mBERT)"
}

def generate_aggregate_report(results_dir="results", plots_dir="results/plots"):
    print("📊 Đang khởi tạo báo cáo tổng hợp (Hệ thống 12 kịch bản)...")
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
            except: continue

    sns.set_style("whitegrid")
    for phase_name, scenario_ids in PHASES.items():
        data_list = []
        for sid in scenario_ids:
            if sid in results:
                data_list.append({"Scenario": sid, "F1-Macro": results[sid], "Technique": LABELS.get(sid, sid)})
        
        if data_list:
            df_phase = pd.DataFrame(data_list)
            plt.figure(figsize=(11, 6))
            palette = "viridis" if "Chặng 4" not in phase_name else "coolwarm"
            ax = sns.barplot(x="Scenario", y="F1-Macro", data=df_phase, palette=palette, hue=None)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                            textcoords='offset points', fontweight='bold')
            plt.title(f"{phase_name}\nComparative Analysis", fontsize=13, fontweight='bold')
            plt.ylim(0.0, 1.0)
            safe_name = phase_name.split(":")[0].replace(" ", "_").lower()
            plt.savefig(os.path.join(plots_dir, f"report_{safe_name}.png"), bbox_inches='tight', dpi=300)
            plt.close()

    # 3. BIỂU ĐỒ TỔNG HỢP TẤT CẢ (Aggregate Performance)
    if results:
        SCENARIO_ORDER = ["S0", "S1", "S1A", "S1B", "S2", "S3", "S4", "S4B", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"]
        all_data = [{"Scenario": sid, "F1-Macro": f1} for sid, f1 in results.items()]
        df_all = pd.DataFrame(all_data)
        df_all["Scenario"] = pd.Categorical(df_all["Scenario"], categories=SCENARIO_ORDER, ordered=True)
        df_all = df_all.sort_values("Scenario").dropna(subset=["Scenario"])
        
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(x="Scenario", y="F1-Macro", data=df_all, palette="viridis", hue="Scenario", legend=False)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                        textcoords='offset points', fontweight='bold')
        
        plt.title("Aggregate Performance Across All Scenarios (F1-Macro)", fontsize=16, fontweight='bold')
        plt.ylim(0.0, 1.05)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(plots_dir, "aggregate_performance.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print("✅ Đã tạo biểu đồ tổng hợp: aggregate_performance.png")

    summary_path = os.path.join(results_dir, "research_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Toàn diện (12 Kịch bản: Multidomain & Multilingual)\n\n")
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
