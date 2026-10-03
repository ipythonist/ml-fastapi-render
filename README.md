# California House Price API

Assignment for the **Getting Started with ML in Production** workshop.
Workflow: **Model → FastAPI → Docker → GitHub → Render → Logging & Monitoring**

A scikit-learn `HistGradientBoostingRegressor` trained on the California housing
dataset (20,640 census blocks, 8 features) predicts the median house value of a
block. The model is saved as `model.pkl` and served by FastAPI.

| Metric (held-out 20%) | Value |
|---|---|
| R² | 0.849 |
| Mean absolute error | ≈ $29,400 |

## Project layout

```
.
├── train.py            # trains the model, writes model.pkl
├── model.pkl           # trained artifact (committed on purpose: Render needs it)
├── main.py             # FastAPI app: /, /health, /predict, /metrics + JSON logging
├── requirements.txt    # pinned runtime deps (same sklearn version trains & serves)
├── Dockerfile          # python:3.11-slim image, honours Render's $PORT
├── render.yaml         # Render Blueprint (docker runtime, free plan, /health check)
├── tests/test_api.py   # pytest suite against the app in-process
└── test_live.py        # smoke test against a running URL (local or Render)
```

## 1. Model

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py          # -> model.pkl
```

## 2. FastAPI (local)

```bash
uvicorn main:app --reload
# open http://127.0.0.1:8000/docs
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
          "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}'
# {"predicted_price_usd": 408146.09, "predicted_value_100k": 4.0815}
```

Run the tests: `pytest -q tests`

| Endpoint | Purpose |
|---|---|
| `GET /` | Service info and model metrics |
| `GET /health` | Liveness check (Render pings this) |
| `POST /predict` | Predict price for one block (8 numeric fields, validated) |
| `GET /metrics` | Request / error / prediction counters and average latency |
| `GET /docs` | Interactive Swagger UI |

## 3. Docker

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

## 4. GitHub

```bash
git add -A && git commit -m "describe change" && git push
```
Render redeploys automatically on every push to `main`.

## 5. Render

Option A, Blueprint (uses `render.yaml`):
1. Render dashboard → **New +** → **Blueprint** → pick this repo → **Apply**.

Option B, manual Web Service:
1. **New +** → **Web Service** → pick this repo.
2. Runtime is auto-detected as **Docker** (because of the Dockerfile). Plan: **Free**.
3. Health check path: `/health`. Click **Create Web Service**.

Then open `https://<your-service>.onrender.com/docs` and try `/predict`.
The free instance sleeps after 15 min idle; the first request can take ~1 min.

## 6. Logging & Monitoring

`main.py` writes one JSON line per request to stdout, which Render shows in the
**Logs** tab of the service:

```
{"ts": "...", "level": "INFO", "event": "model_loaded", "path": "/app/model.pkl", "r2": 0.8486, ...}
{"ts": "...", "level": "INFO", "event": "prediction", "input": {...}, "predicted_price_usd": 408146.09}
{"ts": "...", "level": "INFO", "event": "request", "method": "POST", "path": "/predict", "status": 200, "latency_ms": 12.4}
```

`GET /metrics` returns live counters (requests, errors, predictions, average
latency, average predicted price) for a quick health overview.

Smoke-test the deployed API and capture output for the submission:

```bash
python test_live.py https://<your-service>.onrender.com
```
