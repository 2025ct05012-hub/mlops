"""Train, compare, evaluate, and track two classification models."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import CATEGORICAL, FEATURES, NUMERIC, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/heart_disease.csv"
ARTIFACTS = ROOT / "artifacts"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("numeric", numeric_pipe, NUMERIC), ("categorical", categorical_pipe, CATEGORICAL)])


def model_candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "plots").mkdir(exist_ok=True)
    X, y = load_dataset(DATA)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease-classification")
    results = []
    fitted = {}
    for name, estimator in model_candidates().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"},
            n_jobs=1,
            error_score="raise",
        )
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv", "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
            for metric in ("accuracy", "precision", "recall", "roc_auc"):
                mlflow.log_metric(f"cv_{metric}_mean", float(np.mean(scores[f"test_{metric}"])))
                mlflow.log_metric(f"cv_{metric}_std", float(np.std(scores[f"test_{metric}"])))
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
            }
            for key, value in metrics.items():
                mlflow.log_metric(f"test_{key}", float(value))
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            (ARTIFACTS / f"{name}_classification_report.json").write_text(json.dumps(report, indent=2))
            mlflow.log_artifact(str(ARTIFACTS / f"{name}_classification_report.json"))
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
            ax.set_title(f"{name}: Confusion Matrix")
            fig.tight_layout()
            cm_path = ARTIFACTS / "plots" / f"{name}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=160)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
            ax.set_title(f"{name}: ROC Curve")
            fig.tight_layout()
            roc_path = ARTIFACTS / "plots" / f"{name}_roc_curve.png"
            fig.savefig(roc_path, dpi=160)
            plt.close(fig)
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(
                pipeline, name="model", skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
            )
            results.append(
                {
                    "model": name,
                    **{f"cv_{k}_mean": float(np.mean(scores[f"test_{k}"])) for k in metrics},
                    **{f"test_{k}": float(v) for k, v in metrics.items()},
                    "run_id": run.info.run_id,
                }
            )
            fitted[name] = (pipeline, float(np.mean(scores["test_roc_auc"])))
    results_df = pd.DataFrame(results).sort_values("cv_roc_auc_mean", ascending=False)
    results_df.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    import joblib

    joblib.dump(fitted[best_name][0], ARTIFACTS / "model.joblib")
    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "threshold": 0.5,
        "target_mapping": {"0": "no disease", "1": "disease present"},
        "selection_metric": "mean 5-fold CV ROC-AUC",
        "dataset": "UCI Heart Disease (id 45)",
        "warning": "Educational prototype only; not for clinical use.",
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(results_df.to_string(index=False))
    print(f"Selected {best_name}; model saved to {ARTIFACTS / 'model.joblib'}")


if __name__ == "__main__":
    main()
