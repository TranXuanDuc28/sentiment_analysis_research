from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import re
from datetime import datetime
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thiết lập đường dẫn gốc để import src và api
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.database import save_prediction, get_all_history
from api.crawler import extract_text_from_url

app = FastAPI()

# Cấu hình CORS cho phép frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên giới hạn chỉ localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn model tuyệt đối
MODEL_PATH = os.path.join(parent_dir, "checkpoints", "xlm-roberta-books-en")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading model from: {MODEL_PATH} on device {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
base_model.to(device)
base_model.eval()
print("[INFO] Model loaded successfully!")

# Bộ nhớ đệm lưu lịch sử (fallback khi không có MongoDB)
in_memory_history = []

# Cấu hình WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Khởi tạo ma trận chiếu ngẫu nhiên cố định để phân tích Domain Shift (768 -> 2)
rng = np.random.default_rng(42)
W_proj_np = rng.normal(loc=0.0, scale=1.0, size=(768, 2))
W_proj_np = W_proj_np / np.linalg.norm(W_proj_np, axis=0, keepdims=True)
W_proj = torch.tensor(W_proj_np, dtype=torch.float32).to(device)

# Các câu chuẩn tiếng Anh đại diện cho miền nguồn (Source Domain)
source_sentences = [
    "The character development in this novel was outstanding and kept me reading all night.",
    "A beautifully written piece of literature that explores deep themes of love and loss.",
    "The plot twist at the end was completely unexpected and thrilling.",
    "I found the writing style to be dry and the pacing extremely slow.",
    "This book provides a comprehensive overview of modern history and is highly detailed.",
    "The author describes the setting with such vivid detail, it feels real.",
    "It was a decent read, but the ending felt rushed and unsatisfying.",
    "A masterpiece of science fiction that every fan should read.",
    "I couldn't relate to the main characters and found their decisions frustrating.",
    "The print quality of this hardcover edition is excellent with thick pages.",
    "A classic tale that remains relevant in today's society.",
    "The narration in the audiobook version was engaging and well-paced.",
    "I highly recommend this book to anyone interested in mystery novels.",
    "A disappointing sequel that fails to capture the magic of the first book.",
    "The chapters are short and the story moves along at a fast pace."
]

# Tính toán các điểm nguồn (Source points) tĩnh khi khởi động server
print("[INFO] Preparing source domain projection vectors...")
source_points_raw = []
for s in source_sentences:
    inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = base_model.roberta(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        pooled = outputs.last_hidden_state[:, 0, :]
        proj = torch.matmul(pooled, W_proj).cpu().numpy()[0]
        source_points_raw.append(proj)
source_points_raw = np.array(source_points_raw)
source_centroid = source_points_raw.mean(axis=0)
print("[INFO] Source domain prepared successfully!")

# Helper tính từ quan trọng (Word Importance / Explanations)
def compute_token_importance(text, model, tokenizer, device, pred_class):
    words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    if not words:
        return []
    
    # Giới hạn tối đa 30 từ để xử lý nhanh tránh lag
    words_to_eval = words[:30]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        base_probs = torch.softmax(outputs.logits, dim=-1)
        base_prob = base_probs[0, pred_class].item()
    
    importance_scores = []
    mask_token = tokenizer.mask_token or "<mask>"
    
    for i, word in enumerate(words_to_eval):
        masked_words = words_to_eval.copy()
        masked_words[i] = mask_token
        masked_text = " ".join(masked_words)
        
        m_inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            m_outputs = model(**m_inputs)
            m_probs = torch.softmax(m_outputs.logits, dim=-1)
            m_prob = m_probs[0, pred_class].item()
            
        score = max(0.0, base_prob - m_prob)
        importance_scores.append(score)
        
    # Thêm điểm 0 cho những từ ngoài giới hạn 30
    if len(words) > 30:
        importance_scores.extend([0.0] * (len(words) - 30))
        
    max_score = max(importance_scores) if importance_scores else 0.0
    explanation = []
    for word, score in zip(words, importance_scores):
        norm_score = (score / max_score) if max_score > 0 else 0.0
        explanation.append({
            "word": word,
            "score": round(norm_score, 4)
        })
    return explanation

# Helper phân tích khía cạnh (Aspect Extraction & Sentiment Analysis)
def extract_aspects(text, model, tokenizer, device):
    aspect_keywords = {
        "Price": ["giá", "đắt", "rẻ", "tiền", "chi phí", "thanh toán", "mua", "cost", "price", "expensive", "cheap", "pay", "buy"],
        "Quality": ["chất lượng", "bền", "đẹp", "tốt", "xấu", "kém", "xịn", "hỏng", "lỗi", "chắc chắn", "quality", "material", "durable", "sturdy", "broken", "faulty"],
        "Service": ["phục vụ", "tư vấn", "nhân viên", "thái độ", "chăm sóc", "hỗ trợ", "chu đáo", "nhiệt tình", "service", "staff", "support", "seller", "helpful", "polite"],
        "Delivery": ["giao hàng", "vận chuyển", "ship", "chậm", "nhanh", "đóng gói", "nhận hàng", "delivery", "shipping", "package", "fast", "slow", "received"]
    }
    
    # Chia nhỏ văn bản theo dấu câu
    sentences = re.split(r'[.,?!;]\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if not sentences:
        sentences = [text]
        
    aspect_results = {}
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_aspects = []
        for aspect, keywords in aspect_keywords.items():
            if any(k in sentence_lower for k in keywords):
                matched_aspects.append(aspect)
        
        if matched_aspects:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                pred_class = torch.argmax(outputs.logits, dim=-1).item()
                sentiment = labels[pred_class]
            
            for aspect in matched_aspects:
                if aspect not in aspect_results:
                    aspect_results[aspect] = []
                aspect_results[aspect].append(pred_class)
                
    final_aspects = []
    for aspect, classes in aspect_results.items():
        avg_class = round(sum(classes) / len(classes))
        final_aspects.append({
            "aspect": aspect,
            "sentiment": labels[avg_class]
        })
    return final_aspects

class RequestData(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str

class CompareRequest(BaseModel):
    urls: list[str]

@app.post("/api/predict")
async def predict(data: RequestData):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = base_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
    
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = labels[pred_class]
    
    explanation = compute_token_importance(data.text, base_model, tokenizer, device, pred_class)
    aspects = extract_aspects(data.text, base_model, tokenizer, device)
    
    result = {
        "text": data.text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "aspects": aspects,
        "timestamp": datetime.now().isoformat()
    }
    
    # Lưu vào database
    await save_prediction(result)
    
    # Fallback lưu vào RAM
    in_memory_history.insert(0, result)
    if len(in_memory_history) > 50:
        in_memory_history.pop()
        
    # Phát qua WebSocket để FE cập nhật lịch sử thực tế
    await manager.broadcast({"type": "NEW_ANALYSIS"})
    
    return result

@app.get("/api/history")
async def history():
    db_history = await get_all_history(limit=50)
    if not db_history:
        return in_memory_history
    return db_history

@app.post("/api/analyze-url")
async def analyze_url(data: UrlRequest):
    comments = extract_text_from_url(data.url)
    
    if not comments:
        comments = ["No reviews or comments could be automatically extracted from this link. Please check the URL again."]
        
    results = []
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    for comment in comments[:10]:
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = base_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
            
        explanation = compute_token_importance(comment, base_model, tokenizer, device, pred_class)
        aspects = extract_aspects(comment, base_model, tokenizer, device)
        
        res = {
            "text": comment,
            "sentiment": labels[pred_class],
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "aspects": aspects
        }
        results.append(res)
        
        # Lưu vào lịch sử
        history_item = {
            **res,
            "timestamp": datetime.now().isoformat()
        }
        await save_prediction(history_item)
        in_memory_history.insert(0, history_item)
        
    if len(in_memory_history) > 50:
        in_memory_history = in_memory_history[:50]
        
    await manager.broadcast({"type": "NEW_ANALYSIS"})
    return {"url": data.url, "results": results}

@app.post("/api/compare")
async def compare(data: CompareRequest):
    comparison_data = []
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    for url in data.urls[:2]:
        comments = extract_text_from_url(url)
        if not comments:
            comments = [f"No comments found on {url}"]
            
        results = []
        for comment in comments[:5]:
            inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = base_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_class].item()
                
            results.append({
                "text": comment,
                "sentiment": labels[pred_class],
                "confidence": round(confidence, 4),
                "aspects": extract_aspects(comment, base_model, tokenizer, device)
            })
        comparison_data.append({"url": url, "results": results})
        
    return {"comparison": comparison_data}

@app.post("/api/domain-analysis")
async def domain_analysis(data: RequestData):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = base_model.roberta(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        pooled_target = outputs.last_hidden_state[:, 0, :]
        target_pt_raw = torch.matmul(pooled_target, W_proj).cpu().numpy()[0]
        
    # Tính shift magnitude và similarity score
    shift_magnitude = float(np.linalg.norm(target_pt_raw - source_centroid))
    similarity_score = float(np.exp(-shift_magnitude / 3.0))
    
    # Ghép chung với dữ liệu gốc để chuẩn hóa tỉ lệ vào viewBox [-8, 8] của SVG
    all_pts = np.vstack([source_points_raw, target_pt_raw])
    center = all_pts.mean(axis=0)
    centered_pts = all_pts - center
    max_val = np.max(np.abs(centered_pts))
    scaled_pts = (centered_pts / max_val) * 7.5 if max_val > 0 else centered_pts
    
    source_points = scaled_pts[:-1].tolist()
    target_point = scaled_pts[-1].tolist()
    
    if similarity_score > 0.8:
        status = "Excellent domain alignment. The text matches the source training data distribution perfectly."
    elif similarity_score > 0.5:
        status = "Moderate domain shift detected. The vocabulary or structure deviates slightly from the training domain."
    else:
        status = "Significant domain shift detected (likely due to language change or out-of-domain vocabulary). Prediction confidence may be affected."
        
    return {
        "similarity_score": round(similarity_score, 4),
        "source_points": source_points,
        "target_point": target_point,
        "status": status,
        "shift_magnitude": round(shift_magnitude, 4)
    }

# Endpoint WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
