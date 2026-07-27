# tests/test_price_prediction.py
import pytest
from fastapi.testclient import TestClient
from api.inference_service import app  # your FastAPI app instance


def test_predict_price_success():
    payload = {
    "size_m2": 75.0,
    "nr_of_rooms": 3,
    "floor": 7,
    "building_total_floors": 8,
    "neighbourhood": "Редута",
    "is_first_floor": 0,
    "is_last_floor": 0,
    "is_furnished": 0,
    "near_public_transport": 0,
    }

    with TestClient(app) as client:
        response = client.post("/predictPricePerSqm", json=payload)

    assert response.status_code == 200
    data = response.json()

    print(f"data: {data}")
    assert "log_result" in data
    assert isinstance(data["log_result"], float)
    assert data["log_result"] > 0