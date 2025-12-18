from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd


def get_project_paths() -> Dict[str, Path]:
    """Trả về thư mục gốc project và thư mục models."""
    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / "models"
    return {"root": root_dir, "models": models_dir}


def _prepare_features_from_dict(
    song_features: Dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Chuyển dict thông tin bài hát thành DataFrame một dòng với đúng các cột như lúc train.
    - Nếu có 'release_date' thì tách thành release_year / release_month / release_day.
    - Cột nào thiếu thì điền 0.
    """
    features = dict(song_features)

    if "release_date" in features:
        date = pd.to_datetime(features["release_date"])
        features["release_year"] = int(date.year)
        features["release_month"] = int(date.month)
        features["release_day"] = int(date.day)
        features.pop("release_date")

    df = pd.DataFrame([features])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return df


def predict_streams(song_features: Dict[str, Any]) -> float:
    """
    Dự đoán log_streams bằng mô hình hồi quy baseline.
    Kết quả ở cùng thang với cột log_streams trong dữ liệu (log1p(streams)).
    """
    paths = get_project_paths()
    reg_model_path = paths["models"] / "streams_regressor.pkl"
    artifact = joblib.load(reg_model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    use_log_target = artifact.get("use_log_target", False)
    log_offset = artifact.get("log_offset", None)

    X = _prepare_features_from_dict(song_features, feature_columns)
    y_pred = model.predict(X)

    # Nhánh cũ: trường hợp mô hình tự log-transform target bên trong
    if use_log_target and log_offset is not None:
        y_pred = np.expm1(y_pred) - float(log_offset)

    return float(y_pred[0])


def predict_hit(song_features: Dict[str, Any]) -> Tuple[int, float]:
    """
    Dự đoán Hit (1) / Non-hit (0) bằng mô hình phân loại baseline.
    Trả về (nhãn, xác suất là Hit).
    """
    paths = get_project_paths()
    clf_model_path = paths["models"] / "hit_classifier.pkl"
    artifact = joblib.load(clf_model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X = _prepare_features_from_dict(song_features, feature_columns)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
    else:
        label = int(model.predict(X)[0])
        proba = 1.0 if label == 1 else 0.0
        return label, float(proba)

    label = int(proba >= 0.5)
    return label, float(proba)


def predict_streams_hist_rep(song_features: Dict[str, Any]) -> float:
    """Dự đoán log_streams bằng HistGradientBoostingRegressor (bộ đặc trưng đại diện)."""
    paths = get_project_paths()
    reg_model_path = paths["models"] / "streams_regressor_hist_rep.pkl"
    artifact = joblib.load(reg_model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X = _prepare_features_from_dict(song_features, feature_columns)
    y_pred = model.predict(X)

    return float(y_pred[0])


def predict_hit_stacking(song_features: Dict[str, Any]) -> Tuple[int, float]:
    """
    Dự đoán Hit / Non-hit bằng mô hình Stacking cải tiến
    (base: SVM + RandomForest + HistGradientBoosting, meta: LogisticRegression).
    """
    paths = get_project_paths()
    clf_model_path = paths["models"] / "hit_classifier_stacking.pkl"
    artifact = joblib.load(clf_model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X = _prepare_features_from_dict(song_features, feature_columns)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
    else:
        label = int(model.predict(X)[0])
        proba = 1.0 if label == 1 else 0.0
        return label, float(proba)

    label = int(proba >= 0.5)
    return label, float(proba)


def predict_streams_rep(song_features: Dict[str, Any]) -> float:
    """Alias: dùng mô hình HistGradientBoosting với bộ đặc trưng đại diện."""
    return predict_streams_hist_rep(song_features)


def predict_hit_rep(song_features: Dict[str, Any]) -> Tuple[int, float]:
    """Alias: dùng mô hình Stacking cải tiến."""
    return predict_hit_stacking(song_features)


def compare_streams_predictions(song_features: Dict[str, Any]) -> Dict[str, float | None]:
    """
    So sánh dự đoán hồi quy giữa:
    - baseline (streams_regressor.pkl)
    - HistGradientBoosting với bộ đặc trưng đại diện (streams_regressor_hist_rep.pkl)
    """
    results: Dict[str, float | None] = {
        "baseline_streams": None,
        "histgb_streams": None,
        "difference_histgb_minus_baseline": None,
    }

    try:
        results["baseline_streams"] = predict_streams(song_features)
    except FileNotFoundError:
        results["baseline_streams"] = None

    try:
        results["histgb_streams"] = predict_streams_hist_rep(song_features)
    except FileNotFoundError:
        results["histgb_streams"] = None

    if (
        results["baseline_streams"] is not None
        and results["histgb_streams"] is not None
    ):
        results["difference_histgb_minus_baseline"] = (
            results["histgb_streams"] - results["baseline_streams"]
        )

    return results


def compare_hit_predictions(
    song_features: Dict[str, Any],
) -> Dict[str, int | float | None]:
    """
    So sánh dự đoán Hit / Non-hit giữa:
    - RandomForest baseline (hit_classifier.pkl)
    - Stacking cải tiến (hit_classifier_stacking.pkl)
    """
    results: Dict[str, int | float | None] = {
        "baseline_label": None,
        "baseline_proba": None,
        "stacking_label": None,
        "stacking_proba": None,
    }

    try:
        base_label, base_proba = predict_hit(song_features)
        results["baseline_label"] = base_label
        results["baseline_proba"] = base_proba
    except FileNotFoundError:
        pass

    try:
        stack_label, stack_proba = predict_hit_stacking(song_features)
        results["stacking_label"] = stack_label
        results["stacking_proba"] = stack_proba
    except FileNotFoundError:
        pass

    return results


if __name__ == "__main__":
    # Ví dụ: profile âm thanh + playlist của một bài hit năm 2024.
    # Có thể liên tưởng tới một hit như "Espresso" – Sabrina Carpenter (2024):
    # nhạc pop, danceability cao, xuất hiện nhiều trên playlist và các bảng xếp hạng.
    # Các giá trị số là Z-score tương đối so với data 2023
    # (không phải số liệu thật), mục đích là kiểm tra mô hình với một bài hit mới.
    example_song_name = "Espresso – Sabrina Carpenter (2024, profile giả định)"
    example_song = {
        # artist_count đã được chuẩn hoá; khoảng -0.62 tương ứng ~1 nghệ sĩ chính
        "artist_count": -0.62,
        # các cột playlist / chart sau chuẩn hoá; > 1 ~ top 10% bài hát trong data 2023
        "in_spotify_playlists": 3.0,
        "in_spotify_charts": 3.5,
        "in_apple_playlists": 3.0,
        "in_apple_charts": 3.0,
        "in_deezer_playlists": 2.0,
        "in_deezer_charts": 3.0,
        "in_shazam_charts": 2.5,
        # audio features (Z-score): danceability / energy cao, tempo pop
        "bpm": 1.0,
        "danceability_%": 1.5,
        "valence_%": 0.7,
        "energy_%": 1.3,
        "acousticness_%": -0.5,
        "instrumentalness_%": -0.5,
        "liveness_%": 0.0,
        "speechiness_%": 0.8,
        # 3 đặc trưng đại diện ở mức rất cao
        "artist_avg_streams": 2.5,
        "total_playlists": 2.5,
        "chart_appearances_count": 1.2,
        # key / mode phổ biến của nhạc pop
        "key_A": 0,
        "key_A#": 0,
        "key_B": 0,
        "key_C#": 0,
        "key_D": 0,
        "key_D#": 0,
        "key_E": 1,
        "key_F": 0,
        "key_F#": 0,
        "key_G": 0,
        "key_G#": 0,
        "key_Unknown": 0,
        "mode_Major": 1,
        "mode_Minor": 0,
        # metadata: bài hit phát hành năm 2024
        "release_year": 2024,
        "release_month": 4,
        "release_day": 15,
    }

    print(f"=== Ví dụ bài hát: {example_song_name} ===")

    print("\n=== Hồi quy: baseline vs HistGradientBoosting ===")
    reg_results = compare_streams_predictions(example_song)
    if reg_results["baseline_streams"] is not None:
        print(
            f"- Baseline (streams_regressor.pkl) – dự đoán log_streams: "
            f"{reg_results['baseline_streams']:.3f}"
        )
        approx_streams = np.expm1(reg_results["baseline_streams"])
        print(f"  ≈ Số lượng streams dự đoán (xấp xỉ): {approx_streams:,.0f}")
    else:
        print("- Baseline: chưa tìm thấy model (cần chạy train_model.py)")

    if reg_results["histgb_streams"] is not None:
        print(
            f"- HistGradientBoosting (streams_regressor_hist_rep.pkl) – dự đoán log_streams: "
            f"{reg_results['histgb_streams']:.3f}"
        )
        approx_streams_hist = np.expm1(reg_results["histgb_streams"])
        print(f"  ≈ Số lượng streams dự đoán (xấp xỉ): {approx_streams_hist:,.0f}")
    else:
        print(
            "- HistGradientBoosting: chưa tìm thấy model (cần chạy python src/model_improved.py)"
        )

    print("\n=== Phân loại Hit/Non-hit: baseline vs Stacking ===")
    clf_results = compare_hit_predictions(example_song)
    if clf_results["baseline_label"] is not None:
        print(
            f"- RandomForest baseline (hit_classifier.pkl): "
            f"nhãn={clf_results['baseline_label']}, "
            f"p(Hit)={clf_results['baseline_proba']:.3f}"
        )
    else:
        print("- RandomForest baseline: chưa tìm thấy model (cần chạy train_model.py)")

    if clf_results["stacking_label"] is not None:
        print(
            f"- Stacking cải tiến (hit_classifier_stacking.pkl): "
            f"nhãn={clf_results['stacking_label']}, "
            f"p(Hit)={clf_results['stacking_proba']:.3f}"
        )
    else:
        print(
            "- Stacking cải tiến: chưa tìm thấy model (cần chạy python src/model_improved.py)"
        )

