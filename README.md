# MLOps Assignment 01 — Heart Disease Risk Classification

**Course:** AIMLCZG523 · **Assignment:** 01 · **Dataset:** UCI Heart Disease (Cleveland)

> Educational prototype only. This model is not a medical device and must not be used to diagnose or make treatment decisions.

## Project contents
- `scripts/download_data.py`: fetches the UCI Heart Disease dataset (UCI dataset ID 45) and saves a local CSV.
- `scripts/eda.py`: creates summary statistics and EDA figures.
- `src/train.py`: preprocessing + Logistic Regression and Random Forest, stratified cross-validation, holdout metrics, plots, model serialization, and MLflow logging.
- `src/api.py`: FastAPI `/predict`, `/health`, and Prometheus `/metrics` endpoints with request logging.
- `tests/`: unit and API smoke tests.
- `.github/workflows/ci.yml`: lint, tests, and a training step.
- `Dockerfile`, `docker-compose.yml`: local container run.
- `k8s/`: Kubernetes Deployment, Service, and optional Ingress.
- `report/`: final report document (generated separately in the package).

## Quick start (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/eda.py
python -m src.train
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/docs` for the interactive API documentation.

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age": 58, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236, "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2}'
```

## MLflow
```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Training logs parameters, cross-validation metrics, holdout metrics, confusion matrix, ROC curve, and serialized model artifacts. The best model is selected by mean stratified 5-fold ROC-AUC, then evaluated on a held-out test set.

## Docker
```bash
python scripts/download_data.py
python -m src.train
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```
The model artifact must exist at `artifacts/model.joblib` before building the image. The Dockerfile copies that artifact into the image.

## Kubernetes (local cluster)
Build the image in the same Docker daemon used by Minikube/Docker Desktop, then:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods,svc
```
For Minikube, run `minikube service heart-disease-api --url`. For Docker Desktop, use the service's NodePort at `http://localhost:30080` (if NodePort access is enabled). Ingress is optional and requires an installed ingress controller.

## Reproducibility and caveats
- The dataset download requires network access and the `ucimlrepo` package.
- Results are data- and environment-dependent. Record the actual metrics produced by your run in the report; do not copy example values.
- The CI training job downloads data at runtime. CI needs network access.
- Screenshots, hosted repository URL, public cloud URL, and video are not fabricated. Capture them after you run the workflow/deployment.
- Dataset target is binarized as `0 = no disease`, `1 = disease present` (original target values 1–4 become 1).

## Suggested evidence to capture
1. EDA plots in `artifacts/eda/`.
2. MLflow experiment page showing both runs and metrics.
3. GitHub Actions successful workflow run.
4. Docker container running and `/docs` or `/predict` response.
5. Kubernetes `kubectl get pods,svc` and API response.
6. Prometheus `/metrics` endpoint.
7. A short screen recording showing download → train → test → Docker/API → deployment.

## Academic integrity
Use this as a working scaffold: run the experiments yourself, review the choices, update the report with your actual outputs, and be prepared to explain each component.
