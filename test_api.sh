#!/usr/bin/env bash
# curl walkthrough of every endpoint. Usage: ./test_api.sh [base-url]
BASE="${1:-http://localhost:8000}"
J='Content-Type: application/json'
line() { printf '\n### %s\n' "$1"; }

line "GET /";        curl -s "$BASE/" | python3 -m json.tool
line "GET /health";  curl -s "$BASE/health"; echo
line "POST /predict (valid)"
curl -s -X POST "$BASE/predict" -H "$J" \
  -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}'; echo
line "POST /predict (low-income block)"
curl -s -X POST "$BASE/predict" -H "$J" \
  -d '{"MedInc": 1.5, "HouseAge": 30, "AveRooms": 4.0, "AveBedrms": 1.1, "Population": 2500, "AveOccup": 4.0, "Latitude": 36.5, "Longitude": -119.5}'; echo
line "POST /predict (missing field -> 422)"
curl -s -w '  [HTTP %{http_code}]\n' -X POST "$BASE/predict" -H "$J" -d '{"MedInc": 8.3}'
line "POST /predict (latitude outside California -> 422)"
curl -s -w '  [HTTP %{http_code}]\n' -X POST "$BASE/predict" -H "$J" \
  -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322, "AveOccup": 2.56, "Latitude": 10, "Longitude": -122.23}'
line "GET /metrics";       curl -s "$BASE/metrics" | python3 -m json.tool
line "GET /openapi.json (routes)"; curl -s "$BASE/openapi.json" | python3 -c 'import sys,json; print(list(json.load(sys.stdin)["paths"]))'
