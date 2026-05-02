from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
import sys
from transformers import AutoTokenizer

# Thiết lập đường dẫn gốc để import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.model import DANNModel

app = FastAPI()

# Đường dẫn model tuyệt đối
MODEL_PATH = os.path.join(parent_dir, "checkpoints", "xlm-roberta-books-en")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Khởi tạo model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = DANNModel(num_labels=3)

# Load encoder weights nếu tồn tại
checkpoint_file = os.path.join(MODEL_PATH, "pytorch_model.bin")
if os.path.exists(checkpoint_file):
    print(f"✅ Loading model from: {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.encoder.load_state_dict(state_dict)
else:
    print(f"⚠️ Warning: Không tìm thấy checkpoint tại {checkpoint_file}. Sử dụng mô hình chưa huấn luyện.")

model.to(device)
model.eval()

class RequestData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: RequestData):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        s_logits, _ = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(s_logits, dim=-1).item()
    
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return {
        "text": data.text,
        "sentiment": labels[prediction],
        "label_id": prediction
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
