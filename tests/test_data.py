import pandas as pd
import pytest
from src.data import FEATURES, load_dataset


def test_load_dataset_binarizes_target_and_preserves_features(tmp_path):
    row0 = {c: 1 for c in FEATURES}; row1 = {c: 2 for c in FEATURES}
    df = pd.DataFrame([{**row0, "target_original": 0}, {**row1, "target_original": 2}])
    path = tmp_path / "tiny.csv"; df.to_csv(path, index=False)
    X, y = load_dataset(path)
    assert list(X.columns) == FEATURES
    assert y.tolist() == [0, 1]


def test_missing_required_column_fails(tmp_path):
    path = tmp_path / "bad.csv"; pd.DataFrame({"age": [50]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)
