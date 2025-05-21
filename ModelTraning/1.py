import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

# Đường dẫn tới checkpoint
model_path = Path("C:/Users/Tan/Music/project ban_sach/backend/src/AI/ModelTraning/checkpoint-1488").resolve()

# Load model và tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=str(model_path),
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=str(model_path),
    local_files_only=True
)
model.eval()

# ==== DỮ LIỆU MẪU TEST (3 câu và nhãn tương ứng) ====
test_texts = [
    "Tôi bị đau đầu, mệt mỏi và buồn nôn suốt 3 ngày.",
    "Xuất hiện mẩn đỏ trên da và ngứa khắp người.",
    "Người sốt cao, đau bụng, tiêu chảy nhiều lần."
]
test_labels = [11, 1, 8]  # Giả định: Migraine, Allergy, Gastroenteritis

# Tokenize và dự đoán
encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=1)

# Đánh giá
y_true = test_labels
y_pred = predictions.numpy()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
conf_mat = confusion_matrix(y_true, y_pred)

# In kết quả
print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion Matrix:")
print(conf_mat)
