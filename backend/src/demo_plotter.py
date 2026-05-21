import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def generate_demo_plot(results_dir="results", output_path="results/plots/research_story.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Dữ liệu cho các kịch bản
    demo_data = {
        "S0": 0.963,
        "S1": 0.882, "S1A": 0.811, "S1B": 0.891, "S2": 0.894, "S3": 0.884,
        "S4": 0.785, "S4B": 0.939, "S5": 0.749, "S6": 0.958,
        "S7": 0.832, "S8": 0.905, "S9": 0.861
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
        "1. Multidomain\n(IMDb+Yelp->Amz)": ["S1", "S1A", "S1B", "S2", "S3"],
        "2. Multilingual\n(Anh+Pháp->Việt)": ["S0", "S4", "S4B", "S5", "S6"],
        "3. Unified\n(Full->Việt)": ["S7", "S8", "S9"]
    }
    
    # Custom colors matching the phase themes
    # Phase 1: Teal/Green, Phase 2: Blue/Purple, Phase 3: Orange/Red
    colors = {
        "S1": "#16a085", "S1A": "#1abc9c", "S1B": "#2ecc71", "S2": "#27ae60", "S3": "#1e824c",
        "S0": "#34495e", "S4": "#3498db", "S4B": "#2980b9", "S5": "#9b59b6", "S6": "#8e44ad",
        "S7": "#e67e22", "S8": "#d35400", "S9": "#e74c3c"
    }

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    group_centers = [1.2, 3.6, 5.8]  # Spaced out group centers
    group_names = list(groups.keys())
    
    bar_width = 0.32
    
    legend_handles = []
    
    for g_idx, (group_name, sids) in enumerate(groups.items()):
        center = group_centers[g_idx]
        n_bars = len(sids)
        start_pos = center - ((n_bars - 1) / 2) * bar_width
        
        for i, sid in enumerate(sids):
            pos = start_pos + i * bar_width
            val = demo_data[sid]
            color = colors.get(sid, "#95a5a6")
            
            bar = plt.bar(pos, val, width=bar_width * 0.9, color=color, edgecolor="black", linewidth=0.7)
            legend_handles.append((bar[0], f"{sid}: {val:.3f}"))
            
            # Annotate value above the bar
            plt.annotate(f"{val:.3f}", (pos, val),
                         ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 3),
                         textcoords='offset points', fontweight='bold')
            # Annotate Scenario ID inside the bar
            plt.annotate(sid, (pos, val / 2.0),
                         ha='center', va='center', fontsize=10, color='white',
                         fontweight='bold', rotation=90 if n_bars > 3 else 0)

    plt.title("Tiến trình Hiệu năng Nghiên cứu Toàn diện (F1-Macro)", fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 1.1)
    plt.xticks(group_centers, group_names, fontsize=12, fontweight='bold')
    plt.ylabel("F1-Macro Score", fontsize=12, fontweight='bold')
    
    handles = [item[0] for item in legend_handles]
    labels = [item[1] for item in legend_handles]
    plt.legend(handles, labels, title="Kịch bản", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Đã tạo biểu đồ Research Story tại: {output_path}")

if __name__ == "__main__":
    generate_demo_plot()
