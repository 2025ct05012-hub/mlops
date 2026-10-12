# Suggested 3–5 minute pipeline demo recording

1. Introduce the problem and show the repository structure.
2. Run `python scripts/download_data.py`; show the generated CSV and row/column summary.
3. Run `python scripts/eda.py`; open the class balance and correlation plots.
4. Run `python -m src.train`; show the model comparison CSV and MLflow runs.
5. Run `pytest -q` and `ruff check src scripts tests`.
6. Build and run Docker; show `/health`, `/docs`, one `/predict` request, and `/metrics`.
7. If using Kubernetes, show `kubectl get pods,svc` and call the service endpoint.
8. End with limitations: benchmark dataset only, no clinical use, and no claims beyond the run evidence.

Use actual terminal output in the recording. Do not show credentials, tokens, or private data.
