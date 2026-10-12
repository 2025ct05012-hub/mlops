# Architecture

```mermaid
flowchart LR
  A[UCI Heart Disease dataset] --> B[Download and validation]
  B --> C[EDA and preprocessing]
  C --> D[Stratified CV: Logistic Regression and Random Forest]
  D --> E[MLflow parameters, metrics and artifacts]
  D --> F[Select model by mean CV ROC-AUC]
  F --> G[Serialized preprocessing + model pipeline]
  G --> H[FastAPI /predict]
  H --> I[JSON prediction + probability]
  H --> J[Structured request logs]
  H --> K[Prometheus /metrics]
  L[GitHub Actions] --> M[Lint and tests]
  M --> N[Train and upload artifacts]
  G --> O[Docker image]
  O --> P[Kubernetes Deployment and Service]
```

The training and serving code share a single scikit-learn pipeline, so missing-value imputation, one-hot encoding, scaling, and model inference are packaged together. This reduces training-serving skew. The holdout set is not used for model selection; model selection uses training-set cross-validation.
