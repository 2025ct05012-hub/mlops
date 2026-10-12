"""Download the Heart Disease dataset from the UCI Machine Learning Repository."""

from pathlib import Path
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "heart_disease.csv"


def main() -> None:
    dataset = fetch_ucirepo(id=45)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if targets.shape[1] != 1:
        raise ValueError(f"Expected one target column, found {targets.columns.tolist()}")
    target_name = targets.columns[0]
    data = features.copy()
    data["target_original"] = targets[target_name]
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data)} rows × {len(data.columns)} columns to {OUT}")
    print(f"Original target column: {target_name}; missing cells: {int(data.isna().sum().sum())}")


if __name__ == "__main__":
    main()
