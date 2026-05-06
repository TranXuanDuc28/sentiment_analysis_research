import torch
import yaml
from transformers import AutoTokenizer
from src.dataset import load_amazon_split, load_imdb, load_yelp, make_dataloader
from src.model import BaseModel
from src.visualize_embeddings import visualize_tsne
import os

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # 1. Khởi tạo mô hình và nạp checkpoint S5
    model = BaseModel(config["model"]["name"]).to(device)
    checkpoint_path = "checkpoints/model_s5_multidomain.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy file {checkpoint_path}. Bạn cần chạy Scenario 5 trước.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("🚀 Đã nạp mô hình S5 thành công!")

    # 2. Chuẩn bị dữ liệu để vẽ (Lấy mẫu nhỏ để vẽ cho nhanh)
    print("📦 Đang chuẩn bị dữ liệu...")
    # Nguồn (IMDb + Yelp)
    t1, l1, d1, la1 = load_imdb("test", max_samples=150)
    t2, l2, d2, la2 = load_yelp("test", max_samples=150)
    # Đích (Amazon)
    tt, tl, td, tla = load_amazon_split("english", "all", "test", max_samples=300)
    
    ld_src = make_dataloader(t1+t2, l1+l2, d1+d2, la1+la2, tokenizer, batch_size=32)
    ld_tgt = make_dataloader(tt, tl, td, tla, tokenizer, batch_size=32)

    # 3. Vẽ t-SNE
    print("🎨 Đang vẽ t-SNE cho S5...")
    visualize_tsne(model, tokenizer, [ld_src, ld_tgt], 
                   ["Sources (IMDb+Yelp)", "Target (Amazon)"], 
                   device, "S5_MultiSource_Gap_Before_DANN")
    print("✅ Xong! Kiểm tra file: results/plots/S5_MultiSource_Gap_Before_DANN.png")

if __name__ == "__main__":
    main()
