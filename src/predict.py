from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd


def get_project_paths() -> Dict[str, Path]:
    """Xác định các thư mục chính của project."""
    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / "models"
    return {"root": root_dir, "models": models_dir}


def _prepare_features_from_dict(
    song_features: Dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Chuyển dict input thành DataFrame đúng cột."""
    features = dict(song_features)

    # Nếu người dùng truyền release_date thì tách thành year/month/day
    if "release_date" in features:
        date = pd.to_datetime(features["release_date"])
        features["release_year"] = int(date.year)
        features["release_month"] = int(date.month)
        features["release_day"] = int(date.day)
        features.pop("release_date")

    df = pd.DataFrame([features])

    # Bổ sung các cột thiếu với giá trị 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Chỉ giữ đúng thứ tự cột đã dùng khi train
    df = df[feature_columns]
    return df


def predict_streams(song_features: Dict[str, Any]) -> float:
    """Dự đoán streams (chuẩn hóa) từ features của một bài hát."""
    paths = get_project_paths()
    reg_model_path = paths["models"] / "streams_regressor.pkl"
    artifact = joblib.load(reg_model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    use_log_target = artifact.get("use_log_target", False)
    log_offset = artifact.get("log_offset", None)

    X = _prepare_features_from_dict(song_features, feature_columns)
    y_pred = model.predict(X)

    if use_log_target and log_offset is not None:
        y_pred = np.expm1(y_pred) - float(log_offset)

    return float(y_pred[0])


def predict_hit(song_features: Dict[str, Any]) -> Tuple[int, float]:
    """Dự đoán bài hit (1) / non-hit (0) và xác suất hit."""
    paths = get_project_paths()
    clf_model_path = paths["models"] / "hit_classifier.pkl"
    artifact = joblib.load(clf_model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X = _prepare_features_from_dict(song_features, feature_columns)

    # Lấy xác suất lớp hit (1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
    else:
        # Fallback nếu model không hỗ trợ predict_proba
        label = int(model.predict(X)[0])
        proba = 1.0 if label == 1 else 0.0
        return label, float(proba)

    label = int(proba >= 0.5)
    return label, float(proba)


if __name__ == "__main__":
    # Ví dụ nhỏ cách dùng các hàm dự đoán
    example_song = {
        "artist_count": 1.0,
        "in_spotify_playlists": 50.0,
        "in_spotify_charts": 10.0,
        "in_apple_playlists": 20.0,
        "in_apple_charts": 5.0,
        "in_deezer_playlists": 10.0,
        "in_deezer_charts": 2.0,
        "in_shazam_charts": 3.0,
        "bpm": 120.0,
        "danceability_%": 70.0,
        "valence_%": 60.0,
        "energy_%": 80.0,
        "acousticness_%": 10.0,
        "instrumentalness_%": 0.0,
        "liveness_%": 15.0,
        "speechiness_%": 5.0,
        "artist_avg_streams": 0.0,
        "total_playlists": 80.0,
        "chart_appearances_count": 3.0,
        "key_A": 0,
        "key_A#": 0,
        "key_B": 0,
        "key_C#": 0,
        "key_D": 1,
        "key_D#": 0,
        "key_E": 0,
        "key_F": 0,
        "key_F#": 0,
        "key_G": 0,
        "key_G#": 0,
        "key_Unknown": 0,
        "mode_Major": 1,
        "mode_Minor": 0,
        "release_year": 2023,
        "release_month": 7,
        "release_day": 1,
    }

    predicted_streams = predict_streams(example_song)
    label, proba = predict_hit(example_song)

    print(f"Predicted standardized streams: {predicted_streams:.3f}")
    print(f"Predicted hit label: {label}, hit probability: {proba:.3f}")

