from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

app = FastAPI(title="Network Intrusion Detection API")

Instrumentator().instrument(app).expose(app)

prediction_counter = Counter(
    'predictions_total',
    'Total predictions made',
    ['result']
)

confidence_histogram = Histogram(
    'prediction_confidence',
    'Confidence scores of predictions'
)

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "network-intrusion-classifier"
MODEL_ALIAS = "production"

try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    model_loaded = True
except Exception as e:
    print(f"Failed to load model from registry: {e}")
    model = None
    model_loaded = False


class TrafficFeatures(BaseModel):
    features: dict


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    if not model_loaded:
        return {"error": "Model not loaded"}

    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    return {
        "model_name": MODEL_NAME,
        "version": model_version.version,
        "alias": MODEL_ALIAS,
        "run_id": model_version.run_id,
        "creation_timestamp": model_version.creation_timestamp
    }


@app.post("/predict")
def predict(data: TrafficFeatures):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([data.features])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    result = "ATTACK" if prediction == 1 else "BENIGN"
    confidence = float(max(probability))

    prediction_counter.labels(result=result).inc()
    confidence_histogram.observe(confidence)

    return {
        "prediction": result,
        "confidence": confidence
    }