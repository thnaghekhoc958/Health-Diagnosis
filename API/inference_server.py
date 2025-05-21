from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json

# Đường dẫn đến file ánh xạ
label_file = os.path.join(os.path.dirname(__file__), "label_translations.json")

# Load ánh xạ từ tiếng Anh sang tiếng Việt
with open(label_file, "r", encoding="utf-8") as f:
    label_translations = json.load(f)


app = FastAPI()

# === Load model & tokenizer ===
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ModelTraning")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# === Đọc id2label từ model config và ép kiểu key thành int ===
id2label = {int(k): v for k, v in model.config.id2label.items()}

# === Định nghĩa input dạng JSON gửi đến API ===
class SymptomInput(BaseModel):
    text: str
    language: str = "vi"
# === Endpoint xử lý dự đoán ===
@app.post("/predict")
def predict_symptom(data: SymptomInput):
    print("🔥 Nhận từ NodeJS:", data.dict())  # In toàn bộ JSON đã parse
    inputs = tokenizer(data.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        label_en = id2label[predicted_class]
        
        # Trả kết quả theo ngôn ngữ được truyền vào
        label = label_translations.get(label_en, label_en) if data.language == "vi" else label_en

    return {"label": label}
    
        
