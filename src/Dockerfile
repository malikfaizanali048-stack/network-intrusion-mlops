FROM python:3.11-slim

# cache-bust-v2
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app.py .
COPY dashboard.html .
COPY shap_feature_importance.json .

RUN mkdir -p models && \
    apt-get update && apt-get install -y wget && \
    wget -O models/random_forest_binary.pkl https://github.com/malikfaizanali048-stack/network-intrusion-mlops/releases/download/v1.0-model/random_forest_binary.pkl && \
    apt-get remove -y wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]