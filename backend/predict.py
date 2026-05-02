import torch
from transformers import AutoTokenizer
from src.model import DANNModel
import argparse

def predict_sentiment(text, model_path="checkpoints/xlm-roberta-books-en"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Vì predict chỉ cần inference, ta load encoder đã được lưu
    model = DANNModel(num_labels=3)
    # Load trọng số vào encoder
    state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
    # Lưu ý: Ở đây ta giả định bạn muốn test trên Encoder đã được fine-tune
    # Trong thực tế, bạn nên dùng model hoàn chỉnh nếu muốn test DANN
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        s_logits, _ = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(s_logits, dim=-1).item()

    labels = {0: "Negative 🔴", 1: "Neutral 🟡", 2: "Positive 🟢"}
    return labels[prediction]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Câu văn cần kiểm tra")
    args = parser.parse_args()
    
    result = predict_sentiment(args.text)
    print(f"\nInput: {args.text}")
    print(f"Sentiment: {result}")
