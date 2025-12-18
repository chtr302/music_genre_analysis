# Phân tích Dữ liệu và Dự báo Bài hát Hot trên Spotify 2023

Dự án Khoa học Dữ liệu tập trung phân tích bộ dữ liệu "Most Streamed Spotify Songs 2023" để khám phá các yếu tố tạo nên thành công của một bài hát, đồng thời xây dựng mô hình dự báo lượt nghe (Streams) và hệ thống gợi ý bài hát.

## Nguồn Dữ liệu

*   **Tên bộ dữ liệu:** Most Streamed Spotify Songs 2023
*   **Tác giả (Publisher):** Nidula Elgiriyewithana
*   **Nguồn:** [Kaggle - Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)

Bộ dữ liệu bao gồm các bài hát được nghe nhiều nhất trên Spotify năm 2023, cùng với các thông tin về xếp hạng trên Apple Music, Deezer, Shazam và các đặc trưng âm thanh kỹ thuật (Audio Features).

## Tính năng Chính

1.  **Phân tích dữ liệu (EDA):** Khám phá phân phối lượt nghe, tương quan giữa các nền tảng và các đặc trưng âm thanh.
2.  **Mô hình hóa (Modeling):** Sử dụng kỹ thuật Stacking Ensemble kết hợp Linear Regression, SVM, Random Forest và Gradient Boosting để dự báo lượt stream.
3.  **Phân cụm & Gợi ý:** Phân nhóm bài hát bằng K-Means và gợi ý bài hát tương đồng dựa trên nội dung (Content-based Filtering).
4.  **Dự đoán (Inference):** Module dự đoán đóng gói sẵn, cho phép nhập thông số bài hát và nhận kết quả dự báo.

## Cấu trúc Dự án

├── data/                                                                         
│   ├── spotify-2023.csv           # Dữ liệu gốc                                  
│   ├── spotify_data_processed.csv # Dữ liệu đã làm sạch                          
│   └── data_info.md               # Từ điển dữ liệu                              
├── docs/                          # Tài liệu báo cáo và hướng dẫn                
├── models/                        # Chứa các file model đã huấn luyện (.pkl)     
├── outputs/                       # Biểu đồ và báo cáo kết quả                   
└── src/                                                                          
    ├── analysis/                  # Scripts vẽ biểu đồ phân tích                 
    ├── data_processing/           # Module tiền xử lý, clustering, recommendation
    ├── train_model.py             # Script huấn luyện mô hình                    
    └── predict.py                 # Script chạy dự đoán 

## Hướng dẫn Cài đặt

Yêu cầu môi trường: Python 3.8 trở lên.

Cài đặt các thư viện phụ thuộc:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Hướng dẫn Sử dụng

### 1. Tiền xử lý dữ liệu
Chạy script để làm sạch dữ liệu, xử lý giá trị thiếu và tạo đặc trưng mới:
```bash
python3 src/data_processing/main.py
```

### 2. Huấn luyện Mô hình
Chạy script để huấn luyện toàn bộ các mô hình và lưu kết quả tốt nhất:
```bash
python3 src/train_model.py
```

### 3. Chạy Dự đoán
Thử nghiệm chức năng dự đoán lượt stream và khả năng thành Hit:
```bash
python3 src/predict.py
```

### 4. Hệ thống Gợi ý
Chạy demo hệ thống gợi ý bài hát tương đồng:
```bash
python3 src/data_processing/recommendation/content_based.py
```

## Nhóm Thực hiện
1.  Vũ Phạm Minh Thức - Trưởng nhóm
2.  Trần Công Hậu - Thành viên
3.  Nguyễn Minh Quân - Thành viên

---
Dự án môn học Khoa học Dữ liệu - Học viện Công nghệ Bưu chính Viễn thông.
