from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

prediction_counter = Counter(
    'predictions_total', 
    'Total predictions made', 
    ['result']
)

confidence_histogram = Histogram(
    'prediction_confidence',
    'Confidence scores of predictions'
)

app = FastAPI(title="Network Intrusion Detection API")
Instrumentator().instrument(app).expose(app)

import os

MODEL_PATH = "models/random_forest_binary.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class TrafficFeatures(BaseModel):
    features: dict

@app.get("/health")
def health_check():
    return {"status": "healthy"}

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