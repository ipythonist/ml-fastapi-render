"""Smoke-test a running instance and print evidence for the submission.

Usage:
    python test_live.py                               # local: http://localhost:8000
    python test_live.py https://your-app.onrender.com # deployed on Render
"""
import json
import sys
import requests

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
samples = [
    ("Berkeley hills, high income", {"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
                                     "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}),
    ("Central Valley, low income", {"MedInc": 1.5, "HouseAge": 30, "AveRooms": 4.0, "AveBedrms": 1.1,
                                    "Population": 2500, "AveOccup": 4.0, "Latitude": 36.5, "Longitude": -119.5}),
    ("Los Angeles, mid income", {"MedInc": 4.5, "HouseAge": 25, "AveRooms": 5.2, "AveBedrms": 1.05,
                                 "Population": 1800, "AveOccup": 3.1, "Latitude": 34.05, "Longitude": -118.25}),
]

print(f"Base URL: {base}\n")
r = requests.get(f"{base}/health", timeout=90)  # first call may wake a sleeping free instance
print(f"GET  /health  -> {r.status_code} {r.json()}\n")

for name, payload in samples:
    r = requests.post(f"{base}/predict", json=payload, timeout=90)
    body = r.json()
    print(f"POST /predict -> {r.status_code}  ${body.get('predicted_price_usd'):>12,.0f}   ({name})")

r = requests.get(f"{base}/metrics", timeout=90)
print(f"\nGET  /metrics -> {r.status_code}")
print(json.dumps(r.json(), indent=2))
