import pandas as pd
from src.data import FEATURES
from src.train import build_preprocessor, model_candidates
from sklearn.pipeline import Pipeline


def test_preprocessor_and_models_fit_tiny_dataset():
    rows = []
    for i in range(40):
        rows.append({"age": 30 + i, "sex": i % 2, "cp": i % 4 + 1,
                     "trestbps": 110 + i, "chol": 180 + i, "fbs": i % 2,
                     "restecg": i % 3, "thalach": 180 - i, "exang": i % 2,
                     "oldpeak": (i % 5) / 2, "slope": i % 3 + 1,
                     "ca": i % 4, "thal": i % 3, "target": i % 2})
    df = pd.DataFrame(rows)
    X = df[FEATURES]
    y = df["target"]
    for estimator in model_candidates().values():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        pipe.fit(X, y)
        assert len(pipe.predict(X)) == len(X)
        assert pipe.predict_proba(X).shape == (len(X), 2)
