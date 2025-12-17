## 1. Đánh giá Hiện trạng (Current State Analysis)

- **Thuật toán cũ:** `GradientBoostingRegressor` truyền thống khá chậm trên dữ liệu lớn và không xử lý tốt các giá trị thiếu.
- **Chiến lược tối ưu hóa:** `GridSearchCV` vét cạn (brute-force) tốn rất nhiều thời gian tính toán nhưng chưa chắc tìm được tham số tối ưu toàn cục.
- **Khả năng giải thích:** Chỉ dựa vào `feature_importances_` (của Random Forest) là chưa đủ khách quan, đặc biệt với các đặc trưng có độ phân giải cao.
- **Sự đơn lẻ:** Các mô hình hoạt động độc lập, chưa tận dụng được sức mạnh tổng hợp.

## 2. Chiến lược Cải tiến (Improvement Strategy)

- **Với bài toán Regression**: Chọn Histogram-based Gradient Boosting (HistGradientBoosting) core từ LightGBM nhanh hơn với Gradient hỗ trợ xử lý NaN nội tại luôn. Chủ lực cho bài toán Regress với Class

- **Với bài toán Classification**: Dùng Stacking của Ensemble Learning. Base model (SVM, RF, HistGradient) còn Meta Model là (Linear, Logistic)

### Tối ưu hóa Siêu tham số (Hyperparameter Tuning)
Chuyển từ `GridSearchCV` sang **`RandomizedSearchCV`** hoặc **`HalvingGridSearchCV`**:

### Giải thích Mô hình (Model Explainability - XAI)
Sử dụng **SHAP (SHapley Additive exPlanations)**:
- *Mục tiêu:* Trả lời câu hỏi "Tại sao bài hát này được dự đoán là Hit?".
- *Chi tiết:* SHAP cho biết mỗi đặc trưng (ví dụ: `danceability`) đóng góp bao nhiêu % vào kết quả dự báo cụ thể, thay vì chỉ đưa ra con số chung chung.
