from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="NIDS Prediction API")

model = joblib.load("nids_model_xgboost.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load feature column order from the training data (must match exactly)
FEATURE_COLUMNS = pd.read_csv("cicids2017_downsampled.csv", nrows=1).drop(columns=['Label']).columns.tolist()

class FlowFeatures(BaseModel):
    features: dict  # {feature_name: value, ...}

@app.get("/")
def root():
    return {"status": "NIDS API running", "model": "XGBoost", "classes": len(label_encoder.classes_)}

@app.get("/classes")
def get_classes():
    return {"classes": label_encoder.classes_.tolist()}

@app.post("/predict")
def predict(flow: FlowFeatures):
    row = pd.DataFrame([flow.features])[FEATURE_COLUMNS]
    pred_encoded = model.predict(row)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(row).max()

    return {
        "prediction": pred_label,
        "confidence": round(float(proba), 4)
    }

@app.post("/predict_sample")
def predict_sample():
    """Convenience endpoint: grab a random real row and predict it - for quick demo/testing"""
    df = pd.read_csv("cicids2017_downsampled.csv")
    row = df.sample(n=1)
    true_label = row['Label'].values[0]
    X_row = row[FEATURE_COLUMNS]

    pred_encoded = model.predict(X_row)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(X_row).max()

    return {
        "prediction": pred_label,
        "confidence": round(float(proba), 4),
        "true_label": true_label
    }