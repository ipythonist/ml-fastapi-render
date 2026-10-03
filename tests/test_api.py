from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

RICH_BLOCK = {"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
              "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}
POOR_BLOCK = {"MedInc": 1.5, "HouseAge": 30, "AveRooms": 4.0, "AveBedrms": 1.1,
              "Population": 2500, "AveOccup": 4.0, "Latitude": 36.5, "Longitude": -119.5}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "r2" in r.json()["model_info"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_plausible_price():
    r = client.post("/predict", json=RICH_BLOCK)
    assert r.status_code == 200
    body = r.json()
    assert 100_000 < body["predicted_price_usd"] < 600_000
    assert abs(body["predicted_value_100k"] * 100_000 - body["predicted_price_usd"]) < 10  # 4-dp rounding


def test_richer_block_is_pricier():
    rich = client.post("/predict", json=RICH_BLOCK).json()["predicted_price_usd"]
    poor = client.post("/predict", json=POOR_BLOCK).json()["predicted_price_usd"]
    assert rich > poor


def test_predict_rejects_bad_input():
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={**RICH_BLOCK, "Latitude": 10}).status_code == 422  # outside California
    assert client.post("/predict", json={**RICH_BLOCK, "MedInc": -1}).status_code == 422


def test_metrics_counts_predictions():
    before = client.get("/metrics").json()["predictions_total"]
    client.post("/predict", json=RICH_BLOCK)
    after = client.get("/metrics").json()["predictions_total"]
    assert after == before + 1


def test_dashboard_served():
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "Run all tests" in r.text
