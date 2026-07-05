from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Network Intrusion Detection API")

model = joblib.load("models/random_forest_binary.pkl")

class TrafficFeatures(BaseModel):
    features: dict

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: TrafficFeatures):
    df = pd.DataFrame([data.features])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    result = "ATTACK" if prediction == 1 else "BENIGN"
    confidence = float(max(probability))
    
    return {
        "prediction": result,
        "confidence": confidence
    }