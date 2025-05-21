# Hướng dẫn sử dụng dự án AI

## Giới thiệu
Dự án này là một ứng dụng AI được thiết kế để Chẩn đoán Khám bệnh. Nó cho phép người dùng nhập các triệu chứng cụ thể để chẩn đoán bệnh.

## Yêu cầu
Trước khi bắt đầu, hãy đảm bảo rằng bạn đã cài đặt các thư viện sau:
- Python 3.x
- [Các thư viện cần thiết, ví dụ: NumPy, Pandas, TensorFlow, v.v.]
## bạn có thể sử dụng lệnh này để cài đặt thư viện:
pip install -r requirements.txt

### 🔗 Tải mô hình
Vì giới hạn 100MB của GitHub, model checkpoint không được đẩy trực tiếp lên repo. Bạn có thể tải từ đây:

- [checkpoint-1488 (Google Drive)](https://drive.google.com/drive/folders/1MrNsnjAbis9Cn7HxwsqXzzfjZwaWSxaF?usp=drive_link)
- [checkpoint-1984 (Google Drive)](https://drive.google.com/drive/folders/1eHjWSUpMrrdwaiU7KUUFGMzkQX3vECwC?usp=drive_link)

### Thông tin về các tệp trong dự án
- `generate_symptom_text_dataset.py`: Tệp này được sử dụng để tạo tập dữ liệu văn bản triệu chứng từ các nguồn khác nhau. Nó giúp chuẩn bị dữ liệu cho quá trình huấn luyện mô hình.
- `ModelTranining.py`: Tệp này chứa mã nguồn để huấn luyện mô hình AI. Nó sử dụng các tập dữ liệu đã được chuẩn bị và thực hiện các bước huấn luyện cần thiết.
- `symptom_text_dataset.csv`: Tệp dữ liệu này chứa các triệu chứng được thu thập và định dạng để sử dụng trong quá trình huấn luyện mô hình.
- `vocab.txt`: Tệp này chứa từ vựng được sử dụng trong mô hình, giúp xác định các từ và cụm từ quan trọng trong dữ liệu.


## Cài đặt
1. Tải mã nguồn về máy tính của bạn.
2. Mở terminal hoặc command prompt.
3. Điều hướng đến thư mục dự án:
   ```bash
   cd /Nơi_chứa_Dự_án
   ```
4. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Sử dụng
Để chạy dự án, bạn có thể sử dụng lệnh sau:
```bash
python ModelTranining.py
```
Hoặc nếu bạn muốn chạy một tập lệnh khác, hãy thay thế `ModelTranining.py` bằng tên tệp bạn muốn chạy.



## Liên hệ
Nếu bạn gặp vấn đề hoặc có câu hỏi, hãy liên hệ với [duongletientan2108@gmail.com].
