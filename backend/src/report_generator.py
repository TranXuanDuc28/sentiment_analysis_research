import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Định nghĩa các cụm logic để vẽ biểu đồ (mapping từ ID kịch bản)
CLUSTERS = {
    "Multidomain (S1, S4, S5, S6)": ["S1A", "S4", "S5", "S6A", "S6B"],
    "Multilingual (S1, S2, S3, S12)": ["S1A", "S1B", "S2", "S3A", "S3B", "S12"],
    "Unified Double-Shift (S10, S14)": ["S10A", "S10B", "S14"],
    "Model Comparison (XLM-R vs mBERT)": ["S4", "S11C", "S6B", "S11A", "S2", "S11D", "S10B", "S11B"]
}

def generate_aggregate_report(results_dir="results", plots_dir="results/plots"):
    print("📊 Đang khởi tạo báo cáo tổng hợp kết quả nghiên cứu...")
    os.makedirs(plots_dir, exist_ok=True)
    
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            path = os.path.join(results_dir, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    # Extract ID, e.g. "results_s1a.json" -> "S1A"
                    name_parts = filename.replace("results_", "").replace(".json", "").upper().split("_")
                    key_id = name_parts[0] 
                    results[key_id] = data.get("f1_macro", 0)
            except:
                continue

    if not results:
        print("⚠️ Không tìm thấy file kết quả nào trong thư mục results.")
        return

    # 1. BIỂU ĐỒ THEO CỤM LOGIC (Cluster Plots)
    sns.set_style("whitegrid")
    
    for cluster_name, scenario_ids in CLUSTERS.items():
        data_list = []
        for sid in scenario_ids:
            if sid in results:
                data_list.append({"Scenario": sid, "F1-Macro": results[sid]})
        
        if data_list:
            df_cluster = pd.DataFrame(data_list)
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x="Scenario", y="F1-Macro", data=df_cluster, palette="viridis", hue=None)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, color='black', xytext=(0, 8),
                            textcoords='offset points', fontweight='bold')
            plt.title(f"{cluster_name} Performance", fontsize=15, fontweight='bold')
            plt.ylim(0.0, 1.0) # Scale từ 0 đến 1 để dễ nhìn
            plt.xticks(fontsize=12)
            filename = cluster_name.split()[0].lower() + "_comparison.png"
            plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Đã tạo biểu đồ cụm: {filename}")

    # 2. HEATMAP (Domain Adaptation)
    domains = ["IMDb", "Yelp", "Amazon", "VSFC"]
    heatmap_data = pd.DataFrame(index=domains, columns=domains, dtype=float)

    if "S1A" in results: heatmap_data.loc["IMDb", "IMDb"] = results["S1A"]
    if "S1B" in results: heatmap_data.loc["VSFC", "VSFC"] = results["S1B"]
    if "S7" in results: heatmap_data.loc["Amazon", "Amazon"] = results["S7"]
    if "S4" in results: heatmap_data.loc["IMDb", "Amazon"] = results["S4"]
    if "S2" in results: heatmap_data.loc["IMDb", "VSFC"] = results["S2"]
    if "S6A" in results: heatmap_data.loc["IMDb(DANN)", "Amazon"] = results["S6A"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", annot_kws={"size": 12})
    plt.title("Domain Adaptation Heatmap (F1-Macro)", fontsize=14, fontweight='bold')
    plt.xlabel("Test Domain", fontsize=12)
    plt.ylabel("Train Domain", fontsize=12)
    plt.savefig(os.path.join(plots_dir, "domain_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"🎉 Hoàn tất! Bạn hãy vào '{plots_dir}' để lấy ảnh dán vào Báo Cáo Word.")

if __name__ == "__main__":
    generate_aggregate_report()
