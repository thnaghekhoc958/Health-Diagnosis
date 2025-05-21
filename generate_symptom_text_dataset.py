import pandas as pd

# Load file gốc
df = pd.read_csv("/SymbiPredict/symbipredict_2022.csv")

# Danh sách cột triệu chứng (không gồm cột label - cột bệnh)
symptom_columns = df.columns[:-1]
label_column = df.columns[-1]  # Cột bệnh

# Mapping triệu chứng sang mô tả tiếng Việt
symptom_mapping = {
    "itching": "ngứa",
    "skin_rash": "phát ban da",
    "nodal_skin_eruptions": "nổi hạch trên da",
    "continuous_sneezing": "hắt hơi liên tục",
    "shivering": "rùng mình",
    "chills": "ớn lạnh",
    "joint_pain": "đau khớp",
    "stomach_pain": "đau bụng",
    "acidity": "đầy axit dạ dày",
    "ulcers_on_tongue": "lở miệng",
    "muscle_wasting": "teo cơ",
    "vomiting": "nôn",
    "burning_micturition": "tiểu buốt",
    "spotting_urination": "rối loạn tiểu tiện",
    "fatigue": "mệt mỏi",
    "weight_gain": "tăng cân",
    "anxiety": "lo âu",
    "cold_hands_and_feets": "tay chân lạnh",
    "mood_swings": "thay đổi tâm trạng",
    "weight_loss": "giảm cân",
    "restlessness": "bồn chồn",
    "lethargy": "uể oải",
    "high_fever": "sốt cao",
    "headache": "đau đầu",
    "nausea": "buồn nôn",
    "loss_of_appetite": "mất cảm giác ngon miệng",
}

# Hàm chuyển dòng thành mô tả song ngữ
def convert_row_to_text(row):
    symptoms = []
    for col in symptom_columns:
        if row[col] == 1:
            en = col.replace("_", " ")
            vi = symptom_mapping.get(col, en)
            symptoms.append(f"{en} ({vi})")
    return "Bệnh nhân có các triệu chứng: " + ", ".join(symptoms)

# Tạo DataFrame mới
df_new = pd.DataFrame()
df_new["text"] = df.apply(convert_row_to_text, axis=1)
df_new["label"] = df[label_column]

# Lưu ra file mới
df_new.to_csv("symptom_text_dataset_utf8.csv", index=False, encoding="utf-8")

print("✅ Đã tạo file symptom_text_dataset_utf8.csv thành công.")
