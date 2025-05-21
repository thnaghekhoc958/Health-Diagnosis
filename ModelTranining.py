import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import torch

# === BƯỚC 1: Load dữ liệu ===
csv_path = "symptom_text_dataset.csv"  # File CSV mô tả song ngữ
df = pd.read_csv(csv_path)

# Encode nhãn thành số
labels = df["label"].unique().tolist()
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["label"].map(label2id)

# Tách train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Tạo HuggingFace Datasets
train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})

# === BƯỚC 2: Tokenizer + Model ===
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# === BƯỚC 3: Huấn luyện ===
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.tensor(logits).argmax(dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

# === Cập nhật đường dẫn lưu mô hình theo yêu cầu ===
download_dir = r"C:\Users\1\Music\project ban_sach\nodejs\src\AI\ModelTraning"

# Tạo thư mục nếu chưa có
os.makedirs(download_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=download_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# === BƯỚC 4: Lưu model vào thư mục chỉ định ===
model.save_pretrained(download_dir)
tokenizer.save_pretrained(download_dir)

print(f"Mô hình đã được lưu tại: {download_dir}")
