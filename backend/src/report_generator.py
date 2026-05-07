import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Định nghĩa các pha thực nghiệm theo Narrative mới
PHASES = {
    "Phase 1: Domain Robustness (RQ1)": ["S1", "S2", "S3"],
    "Phase 2: Multilingual Robustness (RQ2)": ["S4", "S5", "S6"],
    "Phase 3: Unified Robustness Framework (RQ3)": ["S7"],
    "Model Comparison (XLM-R vs mBERT)": ["S2", "MBERT_S2", "S4", "MBERT_S4", "S7", "MBERT_S7"]
}

def generate_aggregate_report(results_dir="results", plots_dir="results/plots"):
    print("📊 Đang khởi tạo báo cáo tổng hợp theo Research Narrative mới...")
    os.makedirs(plots_dir, exist_ok=True)
    
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            path = os.path.join(results_dir, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    # Extract ID, e.g. "results_s1.json" -> "S1"
                    name_parts = filename.replace("results_", "").replace(".json", "").upper().split("_")
                    key_id = "_".join(name_parts) # Keep full name parts like MBERT_S4
                    results[key_id] = data.get("f1_macro", 0)
            except:
                continue

    if not results:
        print("⚠️ Không tìm thấy file kết quả nào trong thư mục results.")
        return

    # 1. BIỂU ĐỒ THEO PHA (Phase Plots)
    sns.set_style("whitegrid")
    
    for phase_name, scenario_ids in PHASES.items():
        data_list = []
        for sid in scenario_ids:
            if sid in results:
                data_list.append({"Scenario": sid, "F1-Macro": results[sid]})
        
        if data_list:
            df_phase = pd.DataFrame(data_list)
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="Scenario", y="F1-Macro", data=df_phase, palette="magma", hue=None)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, color='black', xytext=(0, 8),
                            textcoords='offset points', fontweight='bold')
            plt.title(f"{phase_name} Performance", fontsize=15, fontweight='bold')
            plt.ylim(0.0, 1.0)
            plt.xticks(fontsize=12)
            # Safe filename
            safe_name = phase_name.split(":")[0].replace(" ", "_").lower()
            filename = f"{safe_name}_comparison.png"
            plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Đã tạo biểu đồ pha: {filename}")

    # 2. GENERATE RESEARCH SUMMARY MD
    summary_path = os.path.join(results_dir, "research_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Research Findings Summary\n\n")
        for phase_name, scenario_ids in PHASES.items():
            f.write(f"## {phase_name}\n")
            f.write("| Scenario | F1-Macro |\n| :--- | :--- |\n")
            for sid in scenario_ids:
                if sid in results:
                    f.write(f"| {sid} | {results[sid]:.4f} |\n")
            f.write("\n")
    
    print(f"🎉 Hoàn tất! Báo cáo nghiên cứu đã được lưu tại '{summary_path}'")

if __name__ == "__main__":
    generate_aggregate_report()
