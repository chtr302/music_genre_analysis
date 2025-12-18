import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    learning_curve,
    validation_curve,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor


RANDOM_STATE = 42


def get_project_paths() -> Dict[str, Path]:
    """Xác định các thư mục chính của project."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    outputs_dir = root_dir / "outputs"
    models_dir = root_dir / "models"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root_dir,
        "data": data_dir,
        "outputs": outputs_dir,
        "models": models_dir,
    }


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load dataset đã xử lý và tạo thêm label hit."""
    df = pd.read_csv(csv_path)

    # Loại bỏ các hàng thiếu giá trị streams (target)
    df = df.dropna(subset=["streams"])

    # Tách thông tin thời gian từ release_date để dùng như metadata
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_day"] = df["release_date"].dt.day

    # Loại bỏ các cột không hữu ích cho mô hình
    df = df.drop(columns=["track_name", "artist(s)_name", "release_date"])

    # Tạo nhãn hit = 1 nếu thuộc top 10% streams, ngược lại 0
    threshold = df["streams"].quantile(0.9)
    df["hit"] = (df["streams"] >= threshold).astype(int)

    return df


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    """Tạo ColumnTransformer cho numeric và categorical."""
    feature_cols = X.columns.tolist()

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocess, feature_cols


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các metric regression cơ bản."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các metric classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
    }


def train_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    outputs_dir: Path,
) -> Tuple[Dict[Tuple[str, bool], Pipeline], List[Dict[str, Any]]]:
    """Huấn luyện các mô hình regression và (option) log-transform."""
    models: Dict[Tuple[str, bool], Pipeline] = {}
    results: List[Dict[str, Any]] = []

    # 1. Linear Regression
    lin_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    lin_pipe.fit(X_train, y_train)
    y_pred_lin = lin_pipe.predict(X_test)
    metrics_lin = compute_regression_metrics(y_test, y_pred_lin)
    results.append(
        {
            "model": "LinearRegression",
            "use_log_target": False,
            "log_offset": None,
            **metrics_lin,
        }
    )
    models[("LinearRegression", False)] = lin_pipe

    # 2. Ridge Regression + GridSearchCV
    ridge_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ]
    )
    ridge_param_grid = {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
    ridge_grid = GridSearchCV(
        estimator=ridge_pipe,
        param_grid=ridge_param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_
    y_pred_ridge = best_ridge.predict(X_test)
    metrics_ridge = compute_regression_metrics(y_test, y_pred_ridge)
    results.append(
        {
            "model": "Ridge",
            "use_log_target": False,
            "log_offset": None,
            "best_params": ridge_grid.best_params_,
            **metrics_ridge,
        }
    )
    models[("Ridge", False)] = best_ridge

    # 3. Lasso Regression + GridSearchCV
    lasso_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Lasso(random_state=RANDOM_STATE, max_iter=10000)),
        ]
    )
    lasso_param_grid = {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    }
    lasso_grid = GridSearchCV(
        estimator=lasso_pipe,
        param_grid=lasso_param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    lasso_grid.fit(X_train, y_train)
    best_lasso = lasso_grid.best_estimator_
    y_pred_lasso = best_lasso.predict(X_test)
    metrics_lasso = compute_regression_metrics(y_test, y_pred_lasso)
    results.append(
        {
            "model": "Lasso",
            "use_log_target": False,
            "log_offset": None,
            "best_params": lasso_grid.best_params_,
            **metrics_lasso,
        }
    )
    models[("Lasso", False)] = best_lasso

    # 4. Gradient Boosting Regressor (khuyến khích)
    gbr_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )
    gbr_param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    }
    gbr_grid = GridSearchCV(
        estimator=gbr_pipe,
        param_grid=gbr_param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gbr_grid.fit(X_train, y_train)
    best_gbr = gbr_grid.best_estimator_
    y_pred_gbr = best_gbr.predict(X_test)
    metrics_gbr = compute_regression_metrics(y_test, y_pred_gbr)
    results.append(
        {
            "model": "GradientBoostingRegressor",
            "use_log_target": False,
            "log_offset": None,
            "best_params": gbr_grid.best_params_,
            **metrics_gbr,
        }
    )
    models[("GradientBoostingRegressor", False)] = best_gbr

    # 5. Thử log-transform target nếu streams lệch mạnh
    streams_skew = y_train.skew()
    if abs(streams_skew) > 1.0:
        log_offset = float(-y_train.min() + 1e-6)
        y_train_log = np.log1p(y_train + log_offset)

        ridge_log_pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", Ridge(random_state=RANDOM_STATE)),
            ]
        )
        ridge_log_grid = GridSearchCV(
            estimator=ridge_log_pipe,
            param_grid=ridge_param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        ridge_log_grid.fit(X_train, y_train_log)
        best_ridge_log = ridge_log_grid.best_estimator_

        y_pred_ridge_log = best_ridge_log.predict(X_test)
        y_pred_ridge_log_back = np.expm1(y_pred_ridge_log) - log_offset
        metrics_ridge_log = compute_regression_metrics(y_test, y_pred_ridge_log_back)

        results.append(
            {
                "model": "Ridge_log",
                "use_log_target": True,
                "log_offset": log_offset,
                "best_params": ridge_log_grid.best_params_,
                **metrics_ridge_log,
            }
        )
        models[("Ridge", True)] = best_ridge_log

    # Lưu bảng so sánh kết quả regression
    results_df = pd.DataFrame(results)
    results_df.to_csv(outputs_dir / "regression_model_comparison.csv", index=False)

    return models, results


def train_classification_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    outputs_dir: Path,
) -> Tuple[Dict[str, Pipeline], List[Dict[str, Any]]]:
    """Huấn luyện các mô hình classification."""
    models: Dict[str, Pipeline] = {}
    results: List[Dict[str, Any]] = []

    # 1. Naive Bayes (GaussianNB vì feature sau preprocess là liên tục)
    nb_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GaussianNB()),
        ]
    )
    nb_pipe.fit(X_train, y_train)
    y_pred_nb = nb_pipe.predict(X_test)
    metrics_nb = compute_classification_metrics(y_test, y_pred_nb)
    results.append({"model": "GaussianNB", **metrics_nb})
    models["GaussianNB"] = nb_pipe

    # 2. SVM + GridSearchCV
    svm_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", SVC(probability=True, random_state=RANDOM_STATE)),
        ]
    )
    svm_param_grid = [
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"],
        },
        {
            "model__kernel": ["linear"],
            "model__C": [0.1, 1, 10, 100],
        },
    ]
    svm_grid = GridSearchCV(
        estimator=svm_pipe,
        param_grid=svm_param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    y_pred_svm = best_svm.predict(X_test)
    metrics_svm = compute_classification_metrics(y_test, y_pred_svm)
    metrics_svm["best_params"] = svm_grid.best_params_
    results.append({"model": "SVM", **metrics_svm})
    models["SVM"] = best_svm

    # 3. Decision Tree + GridSearchCV
    tree_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )
    tree_param_grid = {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    tree_grid = GridSearchCV(
        estimator=tree_pipe,
        param_grid=tree_param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    tree_grid.fit(X_train, y_train)
    best_tree = tree_grid.best_estimator_
    y_pred_tree = best_tree.predict(X_test)
    metrics_tree = compute_classification_metrics(y_test, y_pred_tree)
    metrics_tree["best_params"] = tree_grid.best_params_
    results.append({"model": "DecisionTree", **metrics_tree})
    models["DecisionTree"] = best_tree

    # 4. Random Forest + GridSearchCV
    rf_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300, n_jobs=-1)),
        ]
    )
    rf_param_grid = {
        "model__max_depth": [5, 10, None],
        "model__max_features": ["sqrt", "log2"],
    }
    rf_grid = GridSearchCV(
        estimator=rf_pipe,
        param_grid=rf_param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    metrics_rf = compute_classification_metrics(y_test, y_pred_rf)
    metrics_rf["best_params"] = rf_grid.best_params_
    results.append({"model": "RandomForest", **metrics_rf})
    models["RandomForest"] = best_rf

    # Lưu bảng so sánh kết quả classification
    results_df = pd.DataFrame(results)
    results_df.to_csv(outputs_dir / "classification_model_comparison.csv", index=False)

    return models, results


def plot_regression_diagnostics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    outputs_dir: Path,
    model_name: str,
) -> None:
    """Vẽ residual plot và predicted vs true."""
    residuals = y_test - y_pred

    # Residual plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted streams (standardized)")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(outputs_dir / f"regression_residuals_{model_name}.png", dpi=150)
    plt.close()

    # Predicted vs True
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("True streams (standardized)")
    plt.ylabel("Predicted streams (standardized)")
    plt.title(f"Predicted vs True - {model_name}")
    plt.tight_layout()
    plt.savefig(outputs_dir / f"regression_pred_vs_true_{model_name}.png", dpi=150)
    plt.close()


def plot_confusion_matrix_and_report(
    best_model_name: str,
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outputs_dir: Path,
) -> None:
    """Vẽ confusion matrix và lưu classification_report."""
    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-hit", "Hit"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(outputs_dir / f"confusion_matrix_{best_model_name}.png", dpi=150)
    plt.close()

    report_str = classification_report(
        y_test,
        y_pred,
        target_names=["Non-hit", "Hit"],
        zero_division=0,
    )
    report_path = outputs_dir / f"classification_report_{best_model_name}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_str)


def analyze_feature_importance_and_coefficients(
    reg_models: Dict[Tuple[str, bool], Pipeline],
    clf_models: Dict[str, Pipeline],
    outputs_dir: Path,
) -> None:
    """Trích xuất feature importance và coefficients top 20."""
    # Random Forest feature importance
    rf_pipeline = clf_models.get("RandomForest")
    if rf_pipeline is not None:
        preprocessor = rf_pipeline.named_steps["preprocess"]
        rf_model: RandomForestClassifier = rf_pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importances = rf_model.feature_importances_

        idx = np.argsort(importances)[::-1][:20]
        top_features = pd.DataFrame(
            {
                "feature": feature_names[idx],
                "importance": importances[idx],
            }
        )
        top_features.to_csv(outputs_dir / "random_forest_feature_importance_top20.csv", index=False)

    # Ridge coefficients
    ridge_pipeline = reg_models.get(("Ridge", False))
    if ridge_pipeline is not None:
        preprocessor = ridge_pipeline.named_steps["preprocess"]
        ridge_model: Ridge = ridge_pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        coefs = ridge_model.coef_.ravel()
        abs_coefs = np.abs(coefs)
        idx = np.argsort(abs_coefs)[::-1][:20]
        ridge_df = pd.DataFrame(
            {
                "feature": feature_names[idx],
                "coefficient": coefs[idx],
                "abs_coefficient": abs_coefs[idx],
            }
        )
        ridge_df.to_csv(outputs_dir / "ridge_coefficients_top20.csv", index=False)

    # Lasso coefficients
    lasso_pipeline = reg_models.get(("Lasso", False))
    if lasso_pipeline is not None:
        preprocessor = lasso_pipeline.named_steps["preprocess"]
        lasso_model: Lasso = lasso_pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        coefs = lasso_model.coef_.ravel()
        abs_coefs = np.abs(coefs)
        idx = np.argsort(abs_coefs)[::-1][:20]
        lasso_df = pd.DataFrame(
            {
                "feature": feature_names[idx],
                "coefficient": coefs[idx],
                "abs_coefficient": abs_coefs[idx],
            }
        )
        lasso_df.to_csv(outputs_dir / "lasso_coefficients_top20.csv", index=False)


def plot_learning_curve_and_validation_curve(
    best_clf_name: str,
    best_clf_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    outputs_dir: Path,
) -> None:
    """Vẽ learning curve cho model tốt nhất và validation curve cho SVM."""
    # Learning curve cho model classification tốt nhất
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=best_clf_model,
        X=X,
        y=y,
        cv=5,
        scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, "o-", label="Train F1-macro")
    plt.plot(train_sizes, val_mean, "o-", label="CV F1-macro")
    plt.xlabel("Training set size")
    plt.ylabel("F1-macro")
    plt.title(f"Learning Curve - {best_clf_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir / f"learning_curve_{best_clf_name}.png", dpi=150)
    plt.close()

    # Validation curve cho SVM theo C
    svm_base = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )
    param_range = np.logspace(-2, 2, 5)
    train_scores_v, val_scores_v = validation_curve(
        estimator=svm_base,
        X=X,
        y=y,
        param_name="model__C",
        param_range=param_range,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    train_mean_v = train_scores_v.mean(axis=1)
    val_mean_v = val_scores_v.mean(axis=1)

    plt.figure(figsize=(6, 4))
    plt.semilogx(param_range, train_mean_v, "o-", label="Train F1-macro")
    plt.semilogx(param_range, val_mean_v, "o-", label="CV F1-macro")
    plt.xlabel("C (SVM)")
    plt.ylabel("F1-macro")
    plt.title("Validation Curve - SVM C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputs_dir / "validation_curve_svm_C.png", dpi=150)
    plt.close()


def save_best_models(
    reg_models: Dict[Tuple[str, bool], Pipeline],
    reg_results: List[Dict[str, Any]],
    clf_models: Dict[str, Pipeline],
    clf_results: List[Dict[str, Any]],
    feature_cols: List[str],
    models_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Chọn và lưu model regression + classification tốt nhất."""
    # Chọn regression model theo RMSE nhỏ nhất
    best_reg_entry = min(reg_results, key=lambda x: x["rmse"])
    best_reg_name = best_reg_entry["model"]
    best_reg_use_log = best_reg_entry.get("use_log_target", False)
    best_reg_log_offset = best_reg_entry.get("log_offset")
    best_reg_model = reg_models[(best_reg_name.split("_")[0], best_reg_use_log)]

    reg_artifact = {
        "model": best_reg_model,
        "model_name": best_reg_name,
        "use_log_target": best_reg_use_log,
        "log_offset": best_reg_log_offset,
        "feature_columns": feature_cols,
    }
    reg_model_path = models_dir / "streams_regressor.pkl"
    joblib.dump(reg_artifact, reg_model_path)

    # Chọn classification model theo F1-macro cao nhất
    best_clf_entry = max(clf_results, key=lambda x: x["f1_macro"])
    best_clf_name = best_clf_entry["model"]
    best_clf_model = clf_models[best_clf_name]

    clf_artifact = {
        "model": best_clf_model,
        "model_name": best_clf_name,
        "feature_columns": feature_cols,
    }
    clf_model_path = models_dir / "hit_classifier.pkl"
    joblib.dump(clf_artifact, clf_model_path)

    return reg_artifact, clf_artifact


def main() -> None:
    """Chạy toàn bộ pipeline train/evaluate/save."""
    paths = get_project_paths()
    data_path = paths["data"] / "spotify_data_processed.csv"
    outputs_dir = paths["outputs"]
    models_dir = paths["models"]

    # 1) Chuẩn bị dữ liệu
    df = load_and_prepare_data(data_path)

    # Không đưa streams/log_streams vào X để tránh leak target
    drop_cols = ["streams", "hit"]
    if "log_streams" in df.columns:
        drop_cols.append("log_streams")

    X = df.drop(columns=drop_cols)
    # Regression target: dùng trực tiếp cột log_streams (log(streams))
    y_reg = df["log_streams"]
    y_clf = df["hit"]

    preprocessor, feature_cols = build_preprocessor(X)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    # 2) Regression models
    reg_models, reg_results = train_regression_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_reg_train,
        y_test=y_reg_test,
        preprocessor=preprocessor,
        outputs_dir=outputs_dir,
    )

    # Chọn model regression tốt nhất cho plot
    best_reg_entry = min(reg_results, key=lambda x: x["rmse"])
    best_reg_name = best_reg_entry["model"]
    best_reg_use_log = best_reg_entry.get("use_log_target", False)
    best_reg_log_offset = best_reg_entry.get("log_offset")
    best_reg_model = reg_models[(best_reg_name.split("_")[0], best_reg_use_log)]

    y_pred_best_reg = best_reg_model.predict(X_test)
    if best_reg_use_log and best_reg_log_offset is not None:
        y_pred_best_reg = np.expm1(y_pred_best_reg) - float(best_reg_log_offset)

    plot_regression_diagnostics(
        y_test=y_reg_test,
        y_pred=y_pred_best_reg,
        outputs_dir=outputs_dir,
        model_name=best_reg_name,
    )

    # 3) Classification models
    clf_models, clf_results = train_classification_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_clf_train,
        y_test=y_clf_test,
        preprocessor=preprocessor,
        outputs_dir=outputs_dir,
    )

    best_clf_entry = max(clf_results, key=lambda x: x["f1_macro"])
    best_clf_name = best_clf_entry["model"]
    best_clf_model = clf_models[best_clf_name]

    plot_confusion_matrix_and_report(
        best_model_name=best_clf_name,
        best_model=best_clf_model,
        X_test=X_test,
        y_test=y_clf_test,
        outputs_dir=outputs_dir,
    )

    # 4) Phân tích sâu mô hình
    analyze_feature_importance_and_coefficients(
        reg_models=reg_models,
        clf_models=clf_models,
        outputs_dir=outputs_dir,
    )

    plot_learning_curve_and_validation_curve(
        best_clf_name=best_clf_name,
        best_clf_model=best_clf_model,
        X=X,
        y=y_clf,
        preprocessor=preprocessor,
        outputs_dir=outputs_dir,
    )

    # 5) Lưu model tốt nhất cho inference
    save_best_models(
        reg_models=reg_models,
        reg_results=reg_results,
        clf_models=clf_models,
        clf_results=clf_results,
        feature_cols=feature_cols,
        models_dir=models_dir,
    )


if __name__ == "__main__":
    main()
