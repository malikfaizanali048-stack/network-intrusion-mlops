FROM python:3.11-slim

# cache-bust-v2
WORKDIR /app
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app.py .
COPY dashboard.html .
COPY models/random_forest_binary.pkl ./models/random_forest_binary.pkl
COPY shap_feature_importance.json .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]