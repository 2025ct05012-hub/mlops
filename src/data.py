"""Data loading and validation helpers."""
from pathlib import Path
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing = sorted(set(FEATURES + ["target_original"]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    X = df[FEATURES].copy().replace("?", pd.NA)
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    target = pd.to_numeric(df["target_original"], errors="coerce")
    valid = target.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = (target.loc[valid] > 0).astype(int).reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError("Expected a binary target with both classes represented")
    return X, y
