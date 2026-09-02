from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction():

    response = client.post(
        "/predict",
        json={
            "prompt": "What is Python?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "risk_score" in data

    assert "action" in data