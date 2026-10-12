from fastapi.testclient import TestClient
from src import api


def test_health_endpoint():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_validates_input():
    client = TestClient(api.app)
    response = client.post("/predict", json={"age": 58})
    assert response.status_code == 422
