.PHONY: install data eda train test lint api
install:
	python -m pip install -r requirements.txt
data:
	python scripts/download_data.py
eda: data
	python scripts/eda.py
train: data
	python -m src.train
test:
	pytest -q
lint:
	ruff check src scripts tests
api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000
