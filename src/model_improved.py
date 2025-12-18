from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)

from train_model import (
    get_project_paths,
    load_and_prepare_data,
    build_preprocessor,
    compute_regression_metrics,
    compute_classification_metrics,
    RANDOM_STATE,
)


# Bo feature dai dien: 3 cot tong hop + mot so audio & metadata
REPRESENTATIVE_FEATURES: List[str] = [
    "artist_avg_streams",
    "total_playlists",
    "chart_appearances_count",
    "bpm",
    "danceability_%",
    "energy_%",
    "valence_%",
    "acousticness_%",
    "speechiness_%",
    "artist_count",
    "release_year",
]


def _load_base_data() -> Dict[str, Any]:
    """Load du lieu da xu ly va tach X_full, y_reg, y_clf."""
    paths = get_project_paths()
    data_path = paths["data"] / "spotify_data_processed.csv"
    df = load_and_prepare_data(data_path)

    # Loại bỏ streams/log_streams và nhãn hit khỏi X_full
    drop_cols = ["streams", "hit"]
    if "log_streams" in df.columns:
        drop_cols.append("log_streams")

    X_full = df.drop(columns=drop_cols)
    # Regression target: log_streams (đã log sẵn trong file processed)
    y_reg = df["log_streams"]
    y_clf = df["hit"]

    return {
        "paths": paths,
        "df": df,
        "X_full": X_full,
        "y_reg": y_reg,
        "y_clf": y_clf,
    }


def run_hist_gradient_boosting_regression() -> None:
    """
    Regression cai tien:
    - Dung HistGradientBoostingRegressor.
    - Train tren bo feature dai dien (3 cot tong hop + audio + metadata).
    - Tuning bang RandomizedSearchCV.
    """
    base = _load_base_data()
    paths = base["paths"]
    df = base["df"]
    y_reg = base["y_reg"]
    y_clf = base["y_clf"]

    outputs_dir: Path = paths["outputs"]
    models_dir: Path = paths["models"]

    # Chi lay bo feature dai dien
    X_rep = df[REPRESENTATIVE_FEATURES].copy()

    # Stratify theo y_clf de giu ti le hit / non-hit
    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X_rep,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, REPRESENTATIVE_FEATURES)]
    )

    base_reg = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_reg),
        ]
    )

    param_distributions = {
        "model__max_depth": [2, 3, 4, None],
        "model__learning_rate": np.linspace(0.01, 0.2, 10),
        "model__max_iter": [100, 200, 300],
        "model__l2_regularization": np.logspace(-3, 1, 5),
        "model__min_samples_leaf": [5, 10, 20, 30],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    search.fit(X_train, y_train)
    best_model: Pipeline = search.best_estimator_

    y_pred = best_model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)

    results = {
        "model": "HistGradientBoostingRegressor_rep",
        "feature_set": "representative_compact",
        "best_params": search.best_params_,
        **metrics,
    }

    results_df = pd.DataFrame([results])
    results_path = outputs_dir / "histgb_regression_representative_features.csv"
    results_df.to_csv(results_path, index=False)

    artifact = {
        "model": best_model,
        "model_name": "HistGradientBoostingRegressor_rep",
        "feature_columns": REPRESENTATIVE_FEATURES,
    }
    model_path = models_dir / "streams_regressor_hist_rep.pkl"
    joblib.dump(artifact, model_path)

    print("Da luu ket qua HistGradientBoosting regression vao:", results_path)
    print("Da luu model HistGradientBoosting regression vao:", model_path)


def run_stacking_classifier() -> None:
    """
    Classification cai tien:
    - Stacking ensemble voi base: SVM, RandomForest, HistGradientBoostingClassifier.
    - Meta-model: LogisticRegression.
    - Tuning bang HalvingGridSearchCV.
    - Dung full feature (bao gom ca 3 cot dai dien playlists/charts/artist).
    """
    base = _load_base_data()
    paths = base["paths"]
    X_full = base["X_full"]
    y_clf = base["y_clf"]

    outputs_dir: Path = paths["outputs"]
    models_dir: Path = paths["models"]

    preprocessor, feature_cols = build_preprocessor(X_full)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_clf,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    # Tinh so luong phan tu it nhat trong cac lop de chon so folds an toan
    unique, counts = np.unique(y_train, return_counts=True)
    min_class_count = int(counts.min())
    # Neu lop hiem chi co 1 mau, khong the stratified > 1 fold
    if min_class_count >= 2:
        cv_folds = min(5, min_class_count)
        cv_param = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
        )
    else:
        # Khong the stratified, dung cv=1 (no CV thuc su, chi de RandomizedSearch chay duoc)
        cv_param = 1

    svc = SVC(probability=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    hist_clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

    base_estimators = [
        ("svc", svc),
        ("rf", rf),
        ("hist", hist_clf),
    ]

    meta_logit = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_logit,
        n_jobs=-1,
        passthrough=False,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", stacking),
        ]
    )

    param_distributions = {
        "model__svc__C": [0.1, 1, 10],
        "model__svc__gamma": ["scale", "auto"],
        "model__rf__max_depth": [5, 10, None],
        "model__hist__max_depth": [None, 3, 5],
        "model__hist__learning_rate": [0.05, 0.1],
        "model__final_estimator__C": [0.1, 1, 10],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1_macro",
        cv=cv_param,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    search.fit(X_train, y_train)

    best_model: Pipeline = search.best_estimator_

    y_pred = best_model.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)

    results = {
        "model": "Stacking_SVC_RF_Hist_metaLogit",
        "feature_set": "full_with_representatives",
        "best_params": search.best_params_,
        **metrics,
    }

    results_df = pd.DataFrame([results])
    results_path = outputs_dir / "stacking_classification_results.csv"
    results_df.to_csv(results_path, index=False)

    artifact = {
        "model": best_model,
        "model_name": "Stacking_SVC_RF_Hist_metaLogit",
        "feature_columns": feature_cols,
    }
    model_path = models_dir / "hit_classifier_stacking.pkl"
    joblib.dump(artifact, model_path)

    print("Da luu ket qua Stacking classification vao:", results_path)
    print("Da luu model Stacking classification vao:", model_path)


def main() -> None:
    """Chay cac model cai tien theo ke hoach improvement."""
    run_hist_gradient_boosting_regression()
    run_stacking_classifier()


if __name__ == "__main__":
    main()
