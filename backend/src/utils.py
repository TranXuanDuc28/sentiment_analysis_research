import torch
import random
import numpy as np
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Lưu state_dict để tiết kiệm dung lượng
    torch.save(model.state_dict(), path)
    print(f"💾 Đã lưu checkpoint tại: {path}")

def load_model_weights(model, path, device="cuda"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"📂 Đã load checkpoint từ: {path}")
        return model
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy checkpoint tại {path}")
        return model

def print_banner(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)
