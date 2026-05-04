
import torch
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import make_dataloader
from transformers import AutoTokenizer
from src.model import BaseModel

def visualize_tsne(model, tokenizer, dataloaders, names, device="cuda", title="t-SNE Embedding Visualization"):
    print(f"🔍 Đang tạo t-SNE cho: {title}...")
    model.eval()
    model.to(device)
    
    all_features = []
    all_groups = []
    
    with torch.no_grad():
        for loader, name in zip(dataloaders, names):
            count = 0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                
                # Trích xuất [CLS] token làm embedding
                outputs = model.encoder(input_ids=input_ids, attention_mask=mask)
                pooled = outputs.last_hidden_state[:, 0, :]
                
                all_features.append(pooled.cpu().numpy())
                all_groups.extend([name] * input_ids.size(0))
                count += input_ids.size(0)
                if count >= 300: break # Giới hạn số điểm để vẽ nhanh và đẹp

    features = np.concatenate(all_features, axis=0)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=all_groups, palette="Set1", alpha=0.7)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results/plots", exist_ok=True)
    safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = f"results/plots/tsne_{safe_title}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu đồ thị tại: {save_path}")

if __name__ == "__main__":
    # Script này có thể chạy độc lập để vẽ đồ thị từ checkpoint
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    from src.dataset import load_amazon_split, load_vsfc
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = BaseModel(config["model"]["name"])
    
    # Ví dụ: Vẽ so sánh EN vs VI
    print("Mẫu: Vẽ t-SNE so sánh không gian đặc trưng Anh - Việt")
    t_en, l_en, d_en = load_amazon_split("english", "books", "test", max_samples=300)
    t_vi, l_vi, d_vi = load_vsfc("test", max_samples=300)
    
    ld_en = make_dataloader(t_en, l_en, d_en, tokenizer, batch_size=16)
    ld_vi = make_dataloader(t_vi, l_vi, d_vi, tokenizer, batch_size=16)
    
    visualize_tsne(model, tokenizer, [ld_en, ld_vi], ["English (Amazon)", "Vietnamese (VSFC)"], device, "Language Alignment Check")
