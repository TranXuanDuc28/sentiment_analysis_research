
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
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    name = filename.replace("results_", "").replace(".json", "").upper()
                    results[name] = data
            except:
                continue

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
    plt.ylim(0.7, 1.0) # Zoom in to see differences clearly
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(plots_dir, "aggregate_performance.png"), bbox_inches='tight')
    plt.close()

    # 2. BẢNG SO SÁNH TỔNG HỢP (Comparison Table)
    print("\n" + "="*80)
    print(f"{'Scenario':<30} | {'Accuracy':<12} | {'F1-Macro':<12}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['Scenario']:<30} | {row['Accuracy']:<12.4f} | {row['F1-Macro']:<12.4f}")
    print("="*80)

    # 3. HEATMAP: Train Domain vs Test Domain (Cross-domain Analysis)
    # Mapping manual cho các kịch bản cross-domain điển hình
    # S1A (IMDb->IMDb), S4 (IMDb->Amz), S8_MDL_YELP (Multi->Yelp), v.v.
    domains = ["IMDb", "Yelp", "Amazon", "VSFC"]
    heatmap_data = pd.DataFrame(index=domains, columns=domains, dtype=float)

    # Fill heatmap from results if they exist
    # Monolingual
    if "S1A" in results: heatmap_data.loc["IMDb", "IMDb"] = results["S1A"]["f1_macro"]
    if "S1B" in results: heatmap_data.loc["VSFC", "VSFC"] = results["S1B"]["f1_macro"]
    if "S7" in results: heatmap_data.loc["Amazon", "Amazon"] = results["S7"]["f1_macro"]
    
    # Transfer
    if "S4" in results: heatmap_data.loc["IMDb", "Amazon"] = results["S4"]["f1_macro"]
    if "S2" in results: heatmap_data.loc["IMDb", "VSFC"] = results["S2"]["f1_macro"]
    
    # DANN
    if "S6A" in results: heatmap_data.loc["IMDb(DANN)", "Amazon"] = results["S6A"]["f1_macro"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Domain Adaptation Heatmap (F1-Macro)")
    plt.xlabel("Test Domain")
    plt.ylabel("Train Domain")
    plt.savefig(os.path.join(plots_dir, "domain_heatmap.png"), bbox_inches='tight')
    plt.close()

    print(f"✅ Đã tạo xong báo cáo tổng hợp tại: {plots_dir}")

if __name__ == "__main__":
    generate_aggregate_report()
