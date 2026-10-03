"""Test every endpoint of a running instance and print a pass/fail summary.

Usage:
    python test_live.py                               # local: http://localhost:8000
    python test_live.py https://your-app.onrender.com # deployed on Render

Exit code is 0 when every check passes, 1 otherwise, so it works in CI too.
"""
import json
import sys
import time
import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
TIMEOUT = 90  # a sleeping Render free instance can take ~1 min to wake up

VALID = {"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
         "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}
SAMPLES = [
    ("Berkeley hills, high income", VALID),
    ("Central Valley, low income", {"MedInc": 1.5, "HouseAge": 30, "AveRooms": 4.0, "AveBedrms": 1.1,
                                    "Population": 2500, "AveOccup": 4.0, "Latitude": 36.5, "Longitude": -119.5}),
    ("Los Angeles, mid income", {"MedInc": 4.5, "HouseAge": 25, "AveRooms": 5.2, "AveBedrms": 1.05,
                                 "Population": 1800, "AveOccup": 3.1, "Latitude": 34.05, "Longitude": -118.25}),
]

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def section(title):
    print(f"\n== {title} ==")


print(f"Base URL: {BASE}")

# 1. GET /
section("GET /  (service info)")
r = requests.get(f"{BASE}/", timeout=TIMEOUT)
body = r.json()
check("status 200", r.status_code == 200, f"got {r.status_code}")
check("lists endpoints", "endpoints" in body, str(body.get("endpoints")))
check("reports model metrics", "r2" in body.get("model_info", {}), f"r2={body.get('model_info', {}).get('r2')}")

# 2. GET /health
section("GET /health")
r = requests.get(f"{BASE}/health", timeout=TIMEOUT)
check("status 200", r.status_code == 200, f"got {r.status_code}")
check("model_loaded is true", r.json().get("model_loaded") is True, json.dumps(r.json()))

# 3. POST /predict, valid inputs
section("POST /predict  (valid inputs)")
prices = {}
for name, payload in SAMPLES:
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/predict", json=payload, timeout=TIMEOUT)
    ms = (time.perf_counter() - t0) * 1000
    body = r.json()
    price = body.get("predicted_price_usd")
    prices[name] = price
    ok = r.status_code == 200 and isinstance(price, (int, float)) and 10_000 < price < 1_000_000
    check(name, ok, f"${price:,.0f}  ({ms:.0f} ms)" if price else json.dumps(body))
check("richer block costs more than poorer block",
      prices["Berkeley hills, high income"] > prices["Central Valley, low income"])

# 4. POST /predict, invalid inputs -> 422
section("POST /predict  (validation errors expected)")
bad_cases = [
    ("empty body", {}),
    ("missing field", {k: v for k, v in VALID.items() if k != "Longitude"}),
    ("latitude outside California", {**VALID, "Latitude": 10}),
    ("negative income", {**VALID, "MedInc": -1}),
    ("string instead of number", {**VALID, "HouseAge": "old"}),
]
for name, payload in bad_cases:
    r = requests.post(f"{BASE}/predict", json=payload, timeout=TIMEOUT)
    check(f"{name} -> 422", r.status_code == 422, f"got {r.status_code}")

# 5. GET /metrics
section("GET /metrics")
r = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
m = r.json()
check("status 200", r.status_code == 200, f"got {r.status_code}")
check("counted our predictions", m.get("predictions_total", 0) >= len(SAMPLES), f"predictions_total={m.get('predictions_total')}")
check("counted our 422s as errors", m.get("errors_total", 0) >= len(bad_cases), f"errors_total={m.get('errors_total')}")
print(json.dumps(m, indent=2))

# 6. GET /docs and /openapi.json
section("GET /docs and /openapi.json")
r = requests.get(f"{BASE}/docs", timeout=TIMEOUT)
check("/docs serves Swagger UI", r.status_code == 200 and "swagger" in r.text.lower())
r = requests.get(f"{BASE}/openapi.json", timeout=TIMEOUT)
paths = list(r.json().get("paths", {}).keys()) if r.status_code == 200 else []
check("/openapi.json lists all routes", {"/", "/health", "/predict", "/metrics"} <= set(paths), str(paths))

# Summary
passed, total = sum(results), len(results)
print(f"\n{'ALL PASSED' if passed == total else 'SOME FAILED'}: {passed}/{total} checks")
sys.exit(0 if passed == total else 1)
