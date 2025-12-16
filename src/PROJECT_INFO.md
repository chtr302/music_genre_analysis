# Giải phẫu hit Spotify 2023 – Mô hình giám sát, tuning và giải thích

Tài liệu này giải thích rõ phần **mô hình hóa giám sát (supervised learning)** trong project, bao gồm:

- Cách xây dựng bài toán Regression và Classification từ dữ liệu Spotify.
- Thiết kế pipeline (train/test split, tiền xử lý trong `Pipeline`/`ColumnTransformer`).
- Các mô hình đã huấn luyện, cách tuning bằng GridSearchCV.
- Cách đánh giá kết quả bằng metric phù hợp.
- Cách giải thích mô hình (feature importance, coefficients).
- Cách lưu model và dùng lại để dự đoán.

Phần xử lý dữ liệu thô và tạo file `spotify_data_processed.csv` được mô tả ở nơi khác, nên ở đây chỉ tập trung vào **model, tuning và interpretation**.

---

## 1. Định nghĩa hai bài toán supervised

### 1.1 Regression – Dự đoán streams

- **Mục tiêu**: dự đoán độ lớn của `streams` (lượt stream), sau khi đã chuẩn hóa (Z-score).
- **Đầu vào (X)**: tất cả các feature trong `spotify_data_processed.csv` ngoại trừ:
  - `streams` – được dùng làm target cho regression.
  - `hit` – nhãn nhị phân cho bài toán classification (sẽ tạo ở bước dưới).
- **Đầu ra (y_reg)**: cột `streams` chuẩn hóa.

Ý tưởng: dựa trên thông tin về playlist, chart, audio features, metadata (năm phát hành, v.v.), mô hình ước lượng “mức độ cao/thấp” của streams cho mỗi bài.

### 1.2 Classification – Dự đoán Hit / Non-hit

- **Mục tiêu**: phân loại mỗi bài hát là Hit (1) hay Non-hit (0) dựa trên streams.
- **Cách tạo nhãn Hit**:
  - Tính ngưỡng top 10%:  
    `threshold = streams.quantile(0.9)`
  - Gán:
    - `hit = 1` nếu `streams >= threshold` → bài nằm trong top 10% streams, coi là “Hit”.
    - `hit = 0` nếu `streams < threshold` → Non-hit.
- **Đầu vào (X)**: cùng tập feature như regression (không dùng `streams` và `hit`).
- **Đầu ra (y_clf)**: cột `hit`.

Lý do chọn quantile 0.9:  
Không dùng một mốc tuyệt đối (ví dụ 1 triệu streams) mà dùng phân vị của dữ liệu thực, để định nghĩa “hit” theo tương quan trong năm đó. Top 10% là mức hợp lý để coi là “thành công nổi bật”.

---

## 2. Thiết kế pipeline supervised

Toàn bộ logic nằm trong `src/train_model.py`.

### 2.1 Chia train/test đúng cách

Sử dụng `train_test_split` với:

- `test_size = 0.2` → 80% train, 20% test.
- `random_state = 42` → kết quả có thể tái lập.
- `stratify = y_clf` → đảm bảo tỷ lệ Hit/Non-hit tương tự giữa train và test.

Lý do:

- Cần một tập test độc lập để đánh giá chất lượng mô hình một cách trung thực.
- Vì Hit chỉ chiếm khoảng 10% nên nếu không stratify, test set có thể quá ít Hit → metric bị méo, không phản ánh đúng khả năng phân loại.

### 2.2 Preprocessing trong `ColumnTransformer` + `Pipeline`

Dù file processed đã được xử lý sơ bộ, pipeline modeling vẫn cần một lớp preprocessing rõ ràng để:

- Đảm bảo mọi bước impute/scale/encode được fit **chỉ trên train** → tránh data leakage.
- Gói toàn bộ logic vào một cấu trúc thống nhất, dễ dùng với GridSearchCV và khi load lại model.

Các bước:

1. Tách feature theo kiểu dữ liệu:
   - `numeric_features`: các cột kiểu `int`/`float`.
   - `categorical_features`: phần còn lại (ví dụ `release_year`, `release_month`, `release_day`, …).

2. Xây dựng pipeline cho numeric:
   - `SimpleImputer(strategy="median")` – điền giá trị thiếu bằng median (ít nhạy với outlier).
   - `StandardScaler()` – chuẩn hóa về mean ~0, std ~1.

3. Xây dựng pipeline cho categorical:
   - `SimpleImputer(strategy="most_frequent")` – điền thiếu bằng giá trị xuất hiện nhiều nhất.
   - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` – mã hóa one-hot, bỏ qua category mới xuất hiện ở test.

4. Gộp lại bằng `ColumnTransformer`:
   - Nhánh `"num"` cho numeric.
   - Nhánh `"cat"` cho categorical.

5. Gắn `ColumnTransformer` với model thành một `Pipeline` duy nhất:
   - `Pipeline([("preprocess", preprocessor), ("model", <Regressor hoặc Classifier>)])`

Khi gọi `fit(X_train, y_train)`, pipeline sẽ:

- Fit imputer + scaler + encoder trên train.
- Fit model trên dữ liệu đã được transform từ train.

Khi gọi `predict(X_test)`:

- Chỉ transform X_test bằng các tham số đã học từ train.
- Sau đó predict → tránh “nhìn trộm” thông tin từ test trong quá trình training.

Đây là cấu trúc chuẩn giúp mô hình ổn định và triển khai lại dễ dàng.

---

## 3. Mô hình Regression – Dự đoán streams

Hàm: `train_regression_models(...)` trong `src/train_model.py`.

### 3.1 Các mô hình sử dụng và lý do

**(1) Linear Regression**

- Mô hình tuyến tính cơ bản:  
  `streams ≈ β0 + β1 x1 + … + βp xp`.
- Mục đích: baseline để so sánh với các mô hình phức tạp hơn.

**(2) Ridge Regression + GridSearchCV**

- Linear Regression với regularization L2 (phạt bình phương độ lớn hệ số).
- Giúp:
  - Giảm overfitting khi có nhiều feature hoặc feature tương quan mạnh.
  - Ổn định hơn so với Linear Regression thuần.
- Dùng `GridSearchCV` để chọn `alpha` tối ưu:
  - Giá trị alpha nhỏ → gần Linear Regression.
  - Alpha lớn → hệ số bị “co lại” nhiều hơn, bias tăng, variance giảm.
- Scoring: `neg_root_mean_squared_error` (tối ưu RMSE).

**(3) Lasso Regression + GridSearchCV**

- Linear Regression với regularization L1.
- Có xu hướng đẩy nhiều hệ số về 0 → tự động **lựa chọn feature**.
- Lý do sử dụng:
  - Dùng Lasso để xem nhóm feature nào thực sự “cốt lõi” cho việc dự đoán streams.
  - Dễ phục vụ phần interpretation (đọc hệ số).

**(4) GradientBoostingRegressor + GridSearchCV**

- Mô hình Boosting kết hợp nhiều cây quyết định nhỏ.
- Mỗi cây học từ residual của cây trước đó → nắm được quan hệ phi tuyến phức tạp.
- Dùng một grid nhỏ (số cây, learning_rate, độ sâu) để:
  - Kiểm soát thời gian train.
  - Vẫn đủ linh hoạt để cải thiện RMSE rõ rệt so với tuyến tính.

### 3.2 Thử log-transform target

Do `streams` lệch phải (skew ~ 2), có thể cân nhắc transform log để:

- Giảm ảnh hưởng của các bài siêu hit (streams cực lớn).
- Làm phân phối target “đẹp” hơn, mô hình tuyến tính dễ fit hơn.

Cách làm:

- Dịch chuyển `y_train` để tránh giá trị ≤ 0, sau đó:
  - `y_train_log = log1p(y_train + offset)`.
- Huấn luyện một phiên bản Ridge trên `y_train_log` với GridSearchCV.
- Khi predict:
  - `y_pred_back = expm1(y_pred_log) - offset` để quay lại thang streams chuẩn hóa.
- So sánh lại R², MAE, RMSE với các model khác.

Kết quả trong project này:

- Phiên bản Ridge_log có RMSE tệ hơn Ridge thường và GradientBoosting.
- Do đó, log-transform **không được chọn** làm phương án cuối, nhưng việc thử và so sánh giúp chứng minh quá trình suy nghĩ và kiểm tra giả thuyết.

### 3.3 Kết quả regression

- Bảng tổng hợp: `outputs/regression_model_comparison.csv`.
- Model tốt nhất (theo RMSE):
  - **GradientBoostingRegressor**
    - R² ≈ 0.87.
    - RMSE ≈ 0.41.
    - MAE ≈ 0.22.
- Các hình vẽ:
  - `regression_residuals_GradientBoostingRegressor.png` – residual plot (sai số so với dự đoán).
  - `regression_pred_vs_true_GradientBoostingRegressor.png` – so sánh predicted vs true với đường y = x.

Hai biểu đồ này giúp kiểm tra:
- Sai số có phân bố ngẫu nhiên hay có pattern (underfit/overfit rõ ràng) không.
- Mức độ bám sát đường lý tưởng y = x của mô hình.

---

## 4. Mô hình Classification – Hit / Non-hit

Hàm: `train_classification_models(...)` trong `src/train_model.py`.

### 4.1 Các mô hình sử dụng và lý do

**(1) Gaussian Naive Bayes (GaussianNB)**

- Giả định mỗi feature (sau preprocessing) có phân phối Gaussian trong từng lớp.
- Đơn giản, tốc độ nhanh, phù hợp làm baseline.

**(2) SVM (Support Vector Machine) + GridSearchCV**

- Ý tưởng: tìm hyperplane phân tách hai lớp với margin lớn nhất.
- Dùng `SVC(probability=True, random_state=42)`.
- Thử hai loại kernel:
  - RBF: bắt quan hệ phi tuyến, check với các giá trị `C` và `gamma` khác nhau.
  - Linear: mô hình tuyến tính nhưng vẫn có khả năng margin tốt.
- Dùng GridSearchCV tối ưu `f1_macro` với `cv=5`, `n_jobs=-1`.

**(3) DecisionTreeClassifier + GridSearchCV**

- Học các rule dạng “if-else” trên feature.
- Dùng grid để chọn:
  - `max_depth`, `min_samples_split`, `min_samples_leaf`.
- Lý do: cây dễ overfit nếu không giới hạn, nên cần tuning để kiểm soát độ phức tạp.

**(4) RandomForestClassifier + GridSearchCV**

- Tập hợp nhiều cây quyết định, mỗi cây train trên một bootstrap sample + subset feature.
- Giảm variance, thường cho kết quả tốt với ít tuning hơn.
- Dùng:
  - `n_estimators = 300`, `random_state = 42`, `n_jobs = -1`.
  - Grid: `max_depth = [5, 10, None]`, `max_features = ["sqrt", "log2"]`.
- Tối ưu theo `f1_macro` với `cv=5`.

### 4.2 Metric đánh giá – đặc biệt F1-macro / F1-micro

Trên tập test, tính các metric:

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `f1_micro`

Giải thích:

- **Accuracy**: tỉ lệ dự đoán đúng trên toàn bộ sample.  
  Không đủ khi dữ liệu mất cân bằng (Non-hit nhiều hơn Hit).
- **Precision / Recall / F1 cho từng lớp**:  
  Cho biết mô hình xử lý riêng từng lớp như thế nào, đặc biệt lớp Hit (lớp quan trọng).
- **F1-macro**:
  - Tính F1 cho từng lớp rồi lấy trung bình.
  - Mỗi lớp được coi quan trọng như nhau → tốt khi muốn cân bằng giữa Hit và Non-hit.
- **F1-micro**:
  - Gộp tất cả TP/FP/FN.
  - Gần với accuracy, bị chi phối bởi lớp có nhiều mẫu.

Trong project, mô hình classification cuối cùng được chọn theo **F1-macro**, vì mục tiêu là nhận diện Hit tốt nhưng không bỏ quên Non-hit.

### 4.3 Kết quả classification

- Bảng tổng hợp: `outputs/classification_model_comparison.csv`.
- Model tốt nhất:
  - **RandomForestClassifier**
    - Accuracy ≈ 0.96.
    - F1-macro ≈ 0.90.
    - F1-micro ≈ 0.96.
    - Best params: `max_depth=10`, `max_features="sqrt"`.
- Hình và báo cáo:
  - `confusion_matrix_RandomForest.png` – ma trận nhầm lẫn Hit/Non-hit.
  - `classification_report_RandomForest.txt` – chi tiết precision, recall, F1 cho từng lớp.

---

## 5. Giải thích mô hình (Interpretation)

### 5.1 Feature importance – RandomForest

File: `outputs/random_forest_feature_importance_top20.csv`.

Những nhóm feature được mô hình đánh giá là quan trọng nhất:

- **Playlist & Chart**:
  - `in_deezer_playlists`, `in_apple_playlists`, `in_spotify_playlists`, `total_playlists`.
  - `in_spotify_charts`, `in_apple_charts`, `in_shazam_charts`, `chart_appearances_count`.
- **Nghệ sĩ**:
  - `artist_avg_streams`.
- **Audio features**:
  - `acousticness_%`, `valence_%`, `energy_%`, `danceability_%`, `speechiness_%`, `bpm`, `liveness_%`.
- **Thời gian**:
  - Một số cột `release_year_*`, `release_day_*`.

Liên hệ với “hit”:

- Bài xuất hiện nhiều trong playlist và chart → có cơ hội được nghe cao hơn → khả năng hit tăng.
- Nghệ sĩ có trung bình streams cao → mỗi lần ra bài mới thường được chú ý nhiều hơn.
- Audio features mô tả mood/energy của bài hát → phù hợp trực giác về các bài hit thường “dễ nghe, dễ nhớ, dễ nhảy”.
- Năm/thời điểm phát hành cũng phản ánh xu hướng âm nhạc từng giai đoạn.

### 5.2 Coefficients – Ridge & Lasso

File:

- `outputs/ridge_coefficients_top20.csv`
- `outputs/lasso_coefficients_top20.csv`

Ý nghĩa:

- **Ridge**:
  - Giữ hầu hết feature, hệ số nhỏ lại.
  - Cho cái nhìn toàn cục về hướng ảnh hưởng: feature nào kéo streams lên, feature nào kéo xuống.
- **Lasso**:
  - Nhiều hệ số bị ép về 0 → chọn ra một nhóm feature “cốt lõi”.

Cách đọc:

- Hệ số dương: tăng feature → streams chuẩn hóa có xu hướng tăng.
- Hệ số âm: tăng feature → streams chuẩn hóa có xu hướng giảm.

Nhìn vào top 20 hệ số, có thể thấy:

- Nhiều feature liên quan tới playlists, charts, artist_avg_streams, total_playlists, release_year rất nổi bật → khớp với feature importance của RandomForest.

---

## 6. Lưu model và hàm dự đoán

Sau khi chọn được mô hình tốt nhất cho regression và classification, bước tiếp theo là đóng gói để dùng lại:

- Regression:
  - Lưu pipeline `GradientBoostingRegressor` (kèm preprocessing) vào `models/streams_regressor.pkl`.
- Classification:
  - Lưu pipeline `RandomForestClassifier` (kèm preprocessing) vào `models/hit_classifier.pkl`.
- Lưu thêm meta:
  - `feature_columns`: danh sách cột đầu vào mà model mong đợi.
  - Thông tin về log-transform target (nếu có dùng).

Trong `src/predict.py`:

- Hàm `predict_streams(song_features: dict) -> float`:
  - Nhận feature của một bài dưới dạng dict.
  - Chuẩn hóa dict thành DataFrame đúng schema (đủ cột, đúng thứ tự).
  - Gọi model regression để dự đoán streams chuẩn hóa (và back-transform nếu có log).

- Hàm `predict_hit(song_features: dict) -> (label: int, proba: float)`:
  - Chuẩn bị input như trên.
  - Gọi model classification để lấy:
    - `label` = 1/0 (Hit / Non-hit).
    - `proba` = xác suất là Hit.

Thiết kế này cho phép:

- Tách rõ bước train (`train_model.py`) và bước dự đoán (`predict.py`).
- Dễ dàng tích hợp model vào notebook khác, API hoặc ứng dụng web sau này.

---

