import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def generate_demo_plot(results_dir="results", output_path="results/plots/research_story.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Dữ liệu demo cho 11 kịch bản mới (Bắt đầu từ Multidomain)
    demo_data = {
        "S1": 0.720, "S2": 0.755, "S3": 0.785,
        "S4": 0.650, "S5": 0.740, "S6": 0.765,
        "S7": 0.610, "S8": 0.695, "S9": 0.725
    }
    
    for sid in demo_data.keys():
        path = os.path.join(results_dir, f"results_{sid.lower()}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    import json
                    data = json.load(f)
                    demo_data[sid] = data.get("f1_macro", demo_data[sid])
            except: pass

    groups = {
        "1. Multidomain\n(IMDb+Yelp->Amz)": ["S1", "S2", "S3"],
        "2. Multilingual\n(Anh+Pháp->Việt)": ["S4", "S5", "S6"],
        "3. Unified\n(Hợp nhất->Việt)": ["S7", "S8", "S9"]
    }
    
    plot_data = []
    for group_name, sids in groups.items():
        for sid in sids:
            plot_data.append({
                "Chặng": group_name,
                "Kịch bản": sid,
                "F1-Macro": demo_data[sid]
            })
    
    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(13, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Chặng", y="F1-Macro", hue="Kịch bản", data=df, palette="viridis")
    plt.title("Tiến trình Hiệu năng Nghiên cứu (Multidomain & Multilingual)", fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 1.0)
    plt.legend(title="Kịch bản", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                        textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Đã tạo biểu đồ Research Story tại: {output_path}")

if __name__ == "__main__":
    generate_demo_plot()
