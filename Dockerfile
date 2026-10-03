# Dockerfile for the workshop assignment API (same shape as the course's day2/Dockerfile)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + trained model
COPY main.py .
COPY model.pkl .
COPY static ./static

# Render injects PORT at runtime; 8000 is the local default
ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
