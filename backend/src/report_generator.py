
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_aggregate_report(results_dir="results", plots_dir="results/plots"):
    print("📊 Đang khởi tạo báo cáo tổng hợp kết quả nghiên cứu...")
    os.makedirs(plots_dir, exist_ok=True)
    
    results = {}
    # Quét tất cả các file kết quả JSON
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            path = os.path.join(results_dir, filename)
            with open(path, "r") as f:
                data = json.load(f)
                name = filename.replace("results_", "").replace(".json", "").upper()
                results[name] = data

    if not results:
        print("⚠️ Không tìm thấy file kết quả nào trong thư mục results.")
        return

    # Chuẩn bị dữ liệu cho biểu đồ tổng quát
    data_list = []
    for name, metrics in results.items():
        data_list.append({
            "Scenario": name,
            "F1-Macro": metrics.get("f1_macro", 0),
            "Accuracy": metrics.get("accuracy", 0)
        })
    
    df = pd.DataFrame(data_list).sort_values("Scenario")

    # 1. BIỂU ĐỒ TỔNG QUÁT: F1-Macro Across Scenarios
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Scenario", y="F1-Macro", data=df, palette="viridis", hue=None)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.title("Aggregate Performance Across All Scenarios (F1-Macro)", fontsize=14)
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(plots_dir, "aggregate_performance.png"), bbox_inches='tight')
    plt.close()

    # 2. BIỂU ĐỒ SO SÁNH: Domain Adaptation (S4a vs S4b)
    # Lưu ý: S4a là baseline, S4b là DANN (thường lưu chung trong kết quả S4)
    # Ở đây chúng ta giả định có S4A và S4B trong results
    if "S4A" in results and "S4" in results:
        comp_df = df[df["Scenario"].isin(["S4A", "S4"])]
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Scenario", y="F1-Macro", data=comp_df, palette="Set2", hue=None)
        plt.title("Domain Adaptation Effect: Baseline vs DANN")
        plt.savefig(os.path.join(plots_dir, "adaptation_effect.png"))
        plt.close()

    # 3. BIỂU ĐỒ SO SÁNH: Cross-lingual Gap (S1B vs S2)
    if "S1B" in results and "S2" in results:
        comp_df = df[df["Scenario"].isin(["S1B", "S2"])]
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Scenario", y="F1-Macro", data=comp_df, palette="coolwarm")
        plt.title("Transfer Gap: VI-Monolingual vs EN-ZeroShot")
        plt.savefig(os.path.join(plots_dir, "transfer_gap.png"))
        plt.close()

    print(f"✅ Đã tạo xong tất cả báo cáo đồ thị tại: {plots_dir}")

if __name__ == "__main__":
    generate_aggregate_report()
